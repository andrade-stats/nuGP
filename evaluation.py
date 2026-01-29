import numpy as np
import sklearn.metrics


def get_marginal_calibration_diagram_data(allData, all_predictive_dist_at_X):

    TOTAL_NR_FOLDS = len(all_predictive_dist_at_X)

    # get smallest values of max-value of each fold
    all_max_values = np.zeros(TOTAL_NR_FOLDS)
    for foldId in range(TOTAL_NR_FOLDS):
        y_cleanTest = allData["all_y_cleanTest"][foldId]
        all_max_values[foldId] = np.max(y_cleanTest)

    min_max = int(np.min(all_max_values))
    print("min_max = ", min_max)
    
    empirical_cdf = np.zeros((TOTAL_NR_FOLDS, min_max + 1))
    predictive_cdf = np.zeros((TOTAL_NR_FOLDS, min_max + 1))

    for foldId in range(TOTAL_NR_FOLDS):

        y_cleanTest = allData["all_y_cleanTest"][foldId]
        pred_dist = all_predictive_dist_at_X[foldId]

        for current_y_value in range(min_max + 1):
            all_probs_all_mc_samples = pred_dist.pmf(k = current_y_value)

            mean_prob = np.mean(all_probs_all_mc_samples)
            
            if current_y_value > 0:
                predictive_cdf[foldId, current_y_value] = predictive_cdf[foldId, current_y_value - 1]
            
            predictive_cdf[foldId, current_y_value] += mean_prob

            empirical_cdf[foldId, current_y_value] = np.sum(y_cleanTest <= current_y_value) / y_cleanTest.shape[0]
    
    return empirical_cdf, predictive_cdf


# calculates SCRPS as proposed in "Local scale invariance and robustness of proper scoring rules", 2023
def get_scaled_crps(allData, all_predictive_dist_at_X, foldId):
    
    NR_MC_SAMPLES = 150

    y_cleanTest = allData["all_y_cleanTest"][foldId]
    pred_dist = all_predictive_dist_at_X[foldId]

    n = y_cleanTest.shape[0]
    GP_SAMPLES = 200

    all_mean_abs_obs = np.zeros((NR_MC_SAMPLES, n))
    all_mean_abs_pairs = np.zeros((NR_MC_SAMPLES, n))

    for mcId in range(NR_MC_SAMPLES):
        if mcId % 100 == 0:
            print("mcd = ", mcId)
        
        rnd_samples_1 = pred_dist.rvs(size=(GP_SAMPLES,n))
        rnd_samples_2 = pred_dist.rvs(size=(GP_SAMPLES,n))
        
        all_mean_abs_obs[mcId] = np.mean(np.abs(rnd_samples_1 - y_cleanTest), axis = 0)
        all_mean_abs_pairs[mcId] = np.mean(np.abs(rnd_samples_1 - rnd_samples_2), axis = 0)

    mean_diff_obs = np.mean(all_mean_abs_obs, axis = 0)
    mean_diff_pairs = np.mean(all_mean_abs_pairs, axis = 0)

    all_scrps = (mean_diff_obs / mean_diff_pairs) + 0.5 * np.log(mean_diff_pairs)
    
    return np.mean(all_scrps)


def getOutlierRecall(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne, calculationForRPrecision = True):
    assert(trueOutlierIndicesZeroOne.shape[0] == estimatedOutlierIndicesZeroOne.shape[0])
    
    nrTrueOutliers = np.sum(trueOutlierIndicesZeroOne)
    
    if calculationForRPrecision:
        assert(np.sum(estimatedOutlierIndicesZeroOne) == nrTrueOutliers)
    
    if nrTrueOutliers == 0:
        return 1.0
    else:
        nrDiscoveredOutliers = np.sum(trueOutlierIndicesZeroOne[estimatedOutlierIndicesZeroOne == 1])
        assert(nrDiscoveredOutliers <= nrTrueOutliers)
        return nrDiscoveredOutliers / nrTrueOutliers

def getNrFalseDetections(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne):
    nrWrongDiscoveries = np.sum(trueOutlierIndicesZeroOne[estimatedOutlierIndicesZeroOne == 1] == 0)
    return nrWrongDiscoveries



def showOutlierDetectionPerformance_power_fdr(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne, dataType = ""):
    
    ROUND_DIGITS = 2
    outlierRecall = getOutlierRecall(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne, calculationForRPrecision = False)
    nrFalseDetections = getNrFalseDetections(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne)
    
    nrDiscoveries = np.sum(estimatedOutlierIndicesZeroOne)
    assert(nrFalseDetections <= nrDiscoveries)
    
    # print("true number of outliers = ", np.sum(trueOutlierIndicesZeroOne))
    # print("estimated number of outliers = ", nrDiscoveries)
    
    if nrDiscoveries == 0:
        FDR = 0.0
    else:
        FDR = nrFalseDetections / nrDiscoveries
    
    # print(dataType + ": outlierRecall(power) = " + str(round(outlierRecall, ROUND_DIGITS)) + ", nrFalseDetections = " + str(nrFalseDetections) + ", FDR = " + str(round(FDR, ROUND_DIGITS)))
    
    return outlierRecall, nrFalseDetections, FDR



def showOutlierDetectionPerformance_auc_top(trueOutlierIndicesZeroOne, pValues, dataType = ""):
    
    ROUND_DIGITS = 2

    nrTrueOutliers = np.sum(trueOutlierIndicesZeroOne)

    if nrTrueOutliers >= 1:
        # calculates R-precision
        assert(np.all(~ np.isnan(pValues)))

        outlierIds = np.argsort(pValues)[0:nrTrueOutliers]
        estimatedOutlierIndicesZeroOne = np.zeros_like(trueOutlierIndicesZeroOne)
        estimatedOutlierIndicesZeroOne[outlierIds] = 1
        outlierRecall_topNrTrueOutliers = getOutlierRecall(trueOutlierIndicesZeroOne, estimatedOutlierIndicesZeroOne)
        
        auc = sklearn.metrics.roc_auc_score(y_true = trueOutlierIndicesZeroOne, y_score = 1.0 - pValues)

        first_outlier_quantile = None
        for nr_from_left, pValue_index in enumerate(np.argsort(-pValues)):
            if trueOutlierIndicesZeroOne[pValue_index] == 1:
                first_outlier_quantile = nr_from_left
                break
        assert(first_outlier_quantile is not None)

        first_outlier_quantile = first_outlier_quantile / trueOutlierIndicesZeroOne.shape[0]
        return auc, outlierRecall_topNrTrueOutliers, first_outlier_quantile
    else:
        print("no outliers therefore auc and outlier recall set to 1.0")
        return 1.0, 1.0, 0


def showAvgAndStd_str(results_allFolds, ROUND_DIGITS = 2):
    m = np.mean(results_allFolds)
    std = np.std(results_allFolds)
    return f"{round(m, ROUND_DIGITS)} ({round(std, ROUND_DIGITS)})"

def showAvgAndStd_percent_str(results_allFolds, ROUND_DIGITS = 2):
    m = np.mean(results_allFolds * 100)
    std = np.std(results_allFolds * 100)
    return f"{round(m, ROUND_DIGITS)}\% ({round(std, ROUND_DIGITS)}\%)"

def showAvgAndStd(results_allFolds, ROUND_DIGITS = 2):
    m = np.mean(results_allFolds)
    std = np.std(results_allFolds)
    return (round(m, ROUND_DIGITS), round(std, ROUND_DIGITS))


def getHighlightedResults(allResult_pairs, bestIsHigh):

    avgResults = np.asarray(allResult_pairs)[:, 0]

    if bestIsHigh is None:
        bestResult = np.nan
    else:
        if bestIsHigh:
            bestResult = np.nanmax(avgResults)
        else:
            bestResult = np.nanmin(avgResults)

    allResultStrs = []

    for avgRes, stdValue in allResult_pairs:
        if np.isnan(avgRes):
            allResultStrs.append("-")
            continue

        resStr = ""
        if avgRes == bestResult:
            resStr += "\\textbf{" + str(avgRes) + "}"
        else:
            resStr += str(avgRes)
        resStr += f" ({stdValue})"
        allResultStrs.append(resStr)

    return " & ".join(allResultStrs)


def showOneLineSummary(allOutlierRecall, allFDR, allNrFalseDetections, auc, outlierRecall_topNrTrueOutliers):
    np.set_printoptions(precision=2)
    print(f"POWER = {np.mean(allOutlierRecall, axis = 1)}, FDR = {np.mean(allFDR, axis = 1)}, nr false dectections = {np.mean(allNrFalseDetections, axis = 1)}, AUC = {np.mean(auc)}, topNrTrueOutlierRecall = {np.mean(outlierRecall_topNrTrueOutliers)}")
    return