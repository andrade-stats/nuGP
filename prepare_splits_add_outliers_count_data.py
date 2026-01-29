
import numpy as np

import simDataGeneration

import commons_data_preparation
import commonSettings

import os

import scipy.stats


def addUniformNoise(y, nrOutliers, rndOutlierIds, outlier_type):
    if nrOutliers == 0:
        return y
    
    y_std = np.std(y)
    CUT_OFFSET = 3.0 * y_std
    OUTLIER_LENGTH = 12.0 * y_std
    
    trueOutlierSamplesRaw = np.random.uniform(low=0.0, high=OUTLIER_LENGTH, size=nrOutliers)
    
    if outlier_type == "symmetric":
        lowerOutliers = -trueOutlierSamplesRaw[trueOutlierSamplesRaw < OUTLIER_LENGTH/2] - CUT_OFFSET 
        higherOutliers = trueOutlierSamplesRaw[trueOutlierSamplesRaw >= OUTLIER_LENGTH/2] - (OUTLIER_LENGTH/2) + CUT_OFFSET
    else:
        assert(outlier_type == "asymmetric_neg" or outlier_type == "asymmetric_pos")
        # scale by 0.5 in order to make symmetric and unsymmetric equally difficult
        lowerOutliers = -trueOutlierSamplesRaw * 0.5 - CUT_OFFSET
        higherOutliers = []
        if outlier_type == "asymmetric_pos":
            # swap
            higherOutliers = -1.0 * lowerOutliers
            lowerOutliers = []

    
    # print("lowerOutliers = ", lowerOutliers)
    # print("higherOutliers = ", higherOutliers)
    
    noise = np.hstack((lowerOutliers, higherOutliers))
    assert(noise.shape[0] == rndOutlierIds.shape[0])
    y[rndOutlierIds] += noise
    return y


def addUniformNoiseMax(y, nrOutliers, outlierIds, outlier_type):
    if nrOutliers == 0:
        return y
    
    if outlier_type == "max1":
        y_max = np.max(y)
        y_std = np.std(y)
        trueOutlierSamplesRaw = y_max + np.random.uniform(low=0.0, high=y_std, size=nrOutliers)
    else:
        assert(False)

    y[outlierIds] = trueOutlierSamplesRaw
    return y

def getNrOutlierAndRndOutlierIds(n, OUTLIER_RATIO):
    nrOutliers = int(n * OUTLIER_RATIO)
    trueOutlierIds = np.arange(n)
    np.random.shuffle(trueOutlierIds)
    trueOutlierIds = trueOutlierIds[0:nrOutliers]
    
    trueOutlierIds_zeroOne = np.zeros(n, dtype = np.int32)
    trueOutlierIds_zeroOne[trueOutlierIds] = 1
    return nrOutliers, trueOutlierIds, trueOutlierIds_zeroOne


# checked
def addNoise_and_scale(X_train, y_train, X_cleanTest, y_cleanTest, TRAINING_DATA_OUTLIER_RATIO, noiseType):

    NORMALIZE_DATA_X = True
    NORMALIZE_DATA_Y = False

    print("y_train = ", y_train)

    # ***************************************************
    nrOutliers, trueOutlierIds, trueOutlierIds_zeroOne = getNrOutlierAndRndOutlierIds(y_train.shape[0], TRAINING_DATA_OUTLIER_RATIO)

    if noiseType.startswith("max"):
        trueOutlierIds = np.argsort(y_train)[0:nrOutliers]
        trueOutlierIds_zeroOne = np.zeros(y_train.shape[0], dtype = np.int32)
        trueOutlierIds_zeroOne[trueOutlierIds] = 1

    if noiseType == "noNoise":
        assert(TRAINING_DATA_OUTLIER_RATIO == 0.0)
        nrOutliers = 0
    elif noiseType == "symmetric":
        y_train = addUniformNoise(y_train, nrOutliers, trueOutlierIds, outlier_type = "symmetric")
    elif noiseType.startswith("asymmetric"):
        y_train = addUniformNoise(y_train, nrOutliers, trueOutlierIds, outlier_type = noiseType)
    elif noiseType.startswith("max"):
        y_train = addUniformNoiseMax(y_train, nrOutliers, trueOutlierIds, outlier_type = noiseType)
    elif noiseType == "focused":
        assert(TRAINING_DATA_OUTLIER_RATIO > 0.0)

        d = X_train.shape[1]
        
        midPoint = int(X_train.shape[0] / 2)

        FOCUS_POINTS_IDS = np.argsort(X_train, axis = 0)[midPoint, :]

        CONCENTRATION_X = scipy.stats.median_abs_deviation(X_train, axis = 0) * 0.1 * d
        jitterX = CONCENTRATION_X * (np.random.uniform(low=0.0, high=1.0, size=(nrOutliers, d)) - 0.5)

        CONCENTRATION_Y = scipy.stats.median_abs_deviation(y_train) * 0.1
        OFFSET_Y = 3.0 * np.std(y_train)
        jitterY = CONCENTRATION_Y * np.random.uniform(low=0.0, high=1.0, size=nrOutliers)
        
        X_train[trueOutlierIds, :] = X_train[FOCUS_POINTS_IDS, np.arange(d)] + jitterX       # median of each dimension + jitter
        y_train[trueOutlierIds] = np.median(y_train[FOCUS_POINTS_IDS]) - OFFSET_Y - jitterY   # median reponse - offset - jitter
    else:
        assert(False)
    # **************************************************
    
    if NORMALIZE_DATA_X:
        X_train, X_cleanTest, dataScalerX = commons_data_preparation.scale_X(X_train, X_cleanTest)
    else:
        dataScalerX = None

    print("y_train = ", y_train)
    if noiseType != "noNoise":
        # make sure that after adding noise the response is still an integer >= 0
        y_train[y_train < 0] = 0
        y_train = np.floor(y_train)
        print("y_train = ", y_train)
    
    assert(np.all(y_train == np.floor(y_train)))
    assert(np.all(y_train >= 0))
    
    if NORMALIZE_DATA_Y:
        y_train, y_cleanTest, dataScalerY = commons_data_preparation.scale_y(y_train, y_cleanTest)
    else:
        dataScalerY = None
    
    
    return X_train, y_train, trueOutlierIds_zeroOne, X_cleanTest, y_cleanTest, dataScalerX, dataScalerY


def splitTrainingAndTest(X, y, testDataRatio):
    
    if testDataRatio > 0.0:
        testDataSize = int(y.shape[0] * testDataRatio)
        
        rndIdOrder = np.arange(y.shape[0])
        np.random.shuffle(rndIdOrder)
        testDataIds = rndIdOrder[0:testDataSize]
        trainDataIds = rndIdOrder[testDataSize:y.shape[0]] 
    
        X_train = X[trainDataIds, :]
        y_train = y[trainDataIds]
        X_test = X[testDataIds, :]
        y_test = y[testDataIds]
    else:
        X_train = X
        y_train = y
        X_test = np.zeros((0,0))
        y_test = np.zeros(0)
    
    return X_train, y_train, X_test, y_test




if __name__ == '__main__':

    if not os.path.exists(commonSettings.PREPARED_DATA_FOLDER):
        os.makedirs(commonSettings.PREPARED_DATA_FOLDER)

    if not os.path.exists(commonSettings.ALL_RESULTS_FOLDER):
        os.makedirs(commonSettings.ALL_RESULTS_FOLDER)

    # "asymmetric_pos" corresponds to "random" in the paper
    # "max1" corresponds to "lowest" in the paper
    ALL_NOISE_TYPES = ["noNoise", "asymmetric_pos", "max1"]

    n = 1000
    kappa = 0.1
    # kappa = 0.5
    datasetName = f"FriedmanCount_n{n}_kappa{kappa}"
    
    # datasetName = "dengue_iquitos"
    # datasetName = "asthma"
    # datasetName = "bike_sharing_hour"
    # datasetName = "bioChemists"
    # datasetName = "NMES"

    # specify here the ratio of outliers
    # TRUE_OUTLIER_RATIO_FOR_NOISE = 0.1
    TRUE_OUTLIER_RATIO_FOR_NOISE = 0.05
    # TRUE_OUTLIER_RATIO_FOR_NOISE = 0.01

    NUMBER_OF_FOLDS = commonSettings.GLOBAL_NUMBER_OF_FOLDS

    for NOISE_TYPE in ALL_NOISE_TYPES:

        np.random.seed(3523421)
        
        # **************************
        if NOISE_TYPE == "noNoise":
            TRUE_OUTLIER_RATIO = 0.0
            noisePostFix = ""
        else:
            TRUE_OUTLIER_RATIO = TRUE_OUTLIER_RATIO_FOR_NOISE
            noisePostFix = "_" + str(int(TRUE_OUTLIER_RATIO * 100))
        # **************************


        print("datasetName = ", datasetName)

        if datasetName.startswith("Friedman"):
            NR_TEST_DATA_SAMPLES = 2000
            
            total_n = n + NR_TEST_DATA_SAMPLES
            allX, allY  = simDataGeneration.getFriedmanCountData(total_n, kappa, nrRepetitions = NUMBER_OF_FOLDS)
        
        else:
            
            allData_original = np.load("openDatasets_prepared/" + datasetName + ".npy", allow_pickle = True).item()
            X_original = allData_original["X_original"]
            y_original = allData_original["y_original"]
            
            allX = []
            allY = []
            
            for foldId in range(NUMBER_OF_FOLDS):
                allX.append(X_original)
                allY.append(y_original)
            
        if datasetName.startswith("Friedman"):
            TEST_DATA_RATIO = NR_TEST_DATA_SAMPLES / total_n
        else:
           TEST_DATA_RATIO = 0.2

        assert(TRUE_OUTLIER_RATIO >= 0.0 and TRUE_OUTLIER_RATIO <= 0.5)

        all_X_train = []
        all_y_train = []
        all_X_cleanTest = []
        all_y_cleanTest = []
        all_trueOutlierIndicesZeroOne = []
        all_dataScalerX = []
        all_dataScalerY = []

        for foldId in range(NUMBER_OF_FOLDS):
            
            print(f"********************** data fold id = {foldId} **********************")

            X_original = allX[foldId]
            y_original = allY[foldId]

            X_original = X_original.astype(np.float32)
            y_original = y_original.astype(np.float32)

            print(f"d = {X_original.shape[1]}, n = {X_original.shape[0]}")
            
            X_cleanTrain, y_cleanTrain, X_cleanTest, y_cleanTest = splitTrainingAndTest(X_original, y_original, TEST_DATA_RATIO)

            X_train, y_train, trueOutlierIndicesZeroOne, X_cleanTest, y_cleanTest, dataScalerX, dataScalerY = addNoise_and_scale(X_cleanTrain, y_cleanTrain, X_cleanTest, y_cleanTest, TRUE_OUTLIER_RATIO, NOISE_TYPE)
            
            print("dataset = ", datasetName)
            print("training data size = ", X_train.shape[0])
            print("test data size = ", X_cleanTest.shape[0])
            
            all_X_train.append(X_train)
            all_y_train.append(y_train)
            all_X_cleanTest.append(X_cleanTest)
            all_y_cleanTest.append(y_cleanTest)
            all_trueOutlierIndicesZeroOne.append(trueOutlierIndicesZeroOne)
            all_dataScalerX.append(dataScalerX)
            all_dataScalerY.append(dataScalerY)

        
        allData = {}
        allData["all_X_train"] = all_X_train
        allData["all_y_train"] = all_y_train
        allData["all_trueOutlierIndicesZeroOne"] = all_trueOutlierIndicesZeroOne
        allData["all_dataScalerX"] = all_dataScalerX
        allData["all_dataScalerY"] = all_dataScalerY
        allData["all_X_cleanTest"] = all_X_cleanTest
        allData["all_y_cleanTest"] = all_y_cleanTest
        np.save(commonSettings.PREPARED_DATA_FOLDER + datasetName + "_" + "trainTestData" + "_" + NOISE_TYPE + noisePostFix,  allData)

        print("*** successfully saved all data ***")

    print("commonSettings.NUMBER_OF_FOLDS = ", NUMBER_OF_FOLDS)