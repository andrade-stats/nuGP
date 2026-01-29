from enum import Enum
import torch
import numpy as np

EstimationType = Enum("EstimationType", ["FULL_INNER_CV", "LIKELIHOOD_NOISE", "SIMPLE_CV_WITH_FINITE_CORRECTION", "SIMPLE_CV_ONLY", "NO_CORRECTION"])

PREPARED_DATA_FOLDER = "openDatasets_prepared/"
ALL_RESULTS_FOLDER = "all_results/"

ALL_BASELINE_METHODS = ["vanilla", "gamma", "student"]

GLOBAL_NUMBER_OF_FOLDS = 10 # 10 should be used for the final experiments


FRIEDMAN_KAPPA01_DATASETS = [("Friedman", 0.1, "noNoise", 0), ("Friedman", 0.1, "asymmetric_pos", 0.05), ("Friedman", 0.1, "asymmetric_pos", 0.1), ("Friedman", 0.1, "max1", 0.05), ("Friedman", 0.1, "max1", 0.1)]
FRIEDMAN_KAPPA05_DATASETS = [("Friedman", 0.5, "noNoise", 0), ("Friedman", 0.5, "asymmetric_pos", 0.05), ("Friedman", 0.5, "asymmetric_pos", 0.1), ("Friedman", 0.5, "max1", 0.05), ("Friedman", 0.5, "max1", 0.1)]
ASTHMA_DATASETS = [("asthma", None, "noNoise", 0), ("asthma", None, "asymmetric_pos", 0.05), ("asthma", None, "asymmetric_pos", 0.1), ("asthma", None, "max1", 0.05), ("asthma", None, "max1", 0.1)]
DENUGE_DATASETS = [("dengue_iquitos", None, "noNoise", 0), ("dengue_iquitos", None, "asymmetric_pos", 0.05), ("dengue_iquitos", None, "asymmetric_pos", 0.1), ("dengue_iquitos", None, "max1", 0.05), ("dengue_iquitos", None, "max1", 0.1)]
BIOCHEMISTS_DATASETS = [("bioChemists", None, "noNoise", 0), ("bioChemists", None, "asymmetric_pos", 0.05), ("bioChemists", None, "asymmetric_pos", 0.1), ("bioChemists", None, "max1", 0.05), ("bioChemists", None, "max1", 0.1)]
NMES_DATASETS = [("NMES", None, "noNoise", 0), ("NMES", None, "asymmetric_pos", 0.05), ("NMES", None, "asymmetric_pos", 0.1), ("NMES", None, "max1", 0.05), ("NMES", None, "max1", 0.1)]
BIKE_DATASETS = [("bike_sharing_hour", None, "noNoise", 0), ("bike_sharing_hour", None, "asymmetric_pos", 0.05), ("bike_sharing_hour", None, "asymmetric_pos", 0.1), ("bike_sharing_hour", None, "max1", 0.05), ("bike_sharing_hour", None, "max1", 0.1)]
ALL_DATASETS = FRIEDMAN_KAPPA01_DATASETS + FRIEDMAN_KAPPA05_DATASETS + ASTHMA_DATASETS + DENUGE_DATASETS + BIOCHEMISTS_DATASETS + NMES_DATASETS + BIKE_DATASETS


# cpu or cuda
DEVICE = None

# data type for tensor 
DATA_TYPE = "float"

# used by proposed method nu-GP
ALL_PRE_SPECFIFIED_NU = np.arange(start = 0.0, stop = 0.21, step=0.01)
TAU = 0.01

# used by proposed method wGP
ALL_W_PRIOR_FAC = [1.0, 10.0, 100.0]


# used by OLRE (values larger than 1.0 were numerically unstable)
ALL_PRIOR_MEDIAN = [0.01, 0.1, 1.0] 

def setDevice():

    global DEVICE

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"

    torch.set_default_device(DEVICE)

    if DATA_TYPE == "double":
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    return

def assertOnDevice(A):
    if DEVICE == "cuda":
        assert(A.is_cuda)
    elif DEVICE == "cpu":
        assert(A.is_cpu)
    else:
        assert(False)

    return


def setDataType(A):

    if DATA_TYPE == "double":
        A = A.double()
    else:
        assert(DATA_TYPE == "float")
        A = A.float()

    if DEVICE == "cuda":
        return A.cuda()
    else:
        return A.cpu()


def getTorchTensor(A):
    if type(A) is float:
        A = torch.tensor([A])
    elif type(A) is np.float32:
        A = torch.tensor(A)
    else:
        A = torch.from_numpy(A)

    A = setDataType(A)
    
    A = A.to(device=DEVICE)
    return A



def get_noise_postfix(noise_type, true_outlier_ratio):
    if noise_type == "noNoise":
        noisePostFix = ""
    else:
        noise_percentage = int(true_outlier_ratio * 100)
        assert(noise_percentage == 10 or noise_percentage == 5 or noise_percentage == 3 or noise_percentage == 1)
        noisePostFix = "_" + str(noise_percentage)

    return noisePostFix



def getLabelName(method):
    if method.startswith("trimmed"):
        return r'$\nu$' + "-GP"
    elif method == "student":
        return r'$t$' + "-GP"
    elif method == "gamma":
        return r'$\gamma$' + "-GP"
    elif method == "vanilla":
        return "GP"
    else:
        assert(False)

def getDatasetName_forPaper(datasetName):
    if datasetName == "Friedman_n100":
        return "F100"
    elif datasetName == "Friedman_n400":
        return "F400"
    elif datasetName == "bodyfat":
        return "body"
    elif datasetName == "housing":
        return "house"
    elif datasetName == "cadata":
        return "cadata"
    elif datasetName == "spacega":
        return "spacega"
    elif datasetName == "syntheticSimpleSin":
        return "bow"
    else:
        assert(False)


def get_all_nu(nu_set):
    ALL_CV_NU_COARSE = [0.3, 0.2, 0.1, 0.01, 0.0]

    ALL_CV_NU_FINE = np.linspace(start=0.3, stop=0.0, num = 11)
    ALL_CV_NU_FINE = [round(v, 2) for v in ALL_CV_NU_FINE]
    
    if nu_set == "coarse":
        assert(False)
        return ALL_CV_NU_COARSE
    if nu_set == "fine":
        return ALL_CV_NU_FINE
    else:
        assert(False)

def get_method_str(args):
    if args.method == "trimmedLB":
        return args.method + "_" + str(args.pre_specified_nu) + "nu"
    if args.method == "trimmedLB_CV" or args.method == "trimmedLB_CV_adv" or args.method == "trimmedLB_inc":
        return args.method + "_" + str(args.pre_specified_nu) + "nu" + "_" + str(args.nr_folds) + "folds"
    elif args.method == "trimmedLB_opt_final_trim_eval":
        return args.method + "_" + str(args.pre_specified_nu) + "nu" + "_" + args.nu_set
    elif args.method == "trimmedLB_opt_final_mean_eval":
        return args.method + "_" + args.nu_set
    elif args.method == "trimmedLB_refineNu":
        assert(args.nu_selection_method is not None)
        return args.method + "_" + str(args.pre_specified_nu) + "nu" + "_" + args.nu_selection_method
    elif args.method == "gammaDivergence":
        return args.method + "_" + str(args.gamma) + "gamma"
    elif args.method.startswith("OLRE"):
        return args.method + "_" + str(args.prior_median) + "priormed"
    elif args.method == "wGP":
        return args.method + "_" + str(args.wl_prior_fac) + "prior"
    elif args.method == "wGP_CV":
        return args.method + "_" + str(args.wl_prior_fac) + "prior" + "_" + str(args.nr_folds) + "folds"
    elif args.method == "wGP_trimmed":
        return args.method + "_" + str(args.wl_prior_fac) + "prior" + "_" + str(args.pre_specified_nu) + "nu"
    elif (args.method == "variationalApprox" or args.method == "gamma_divergence_CV") and args.likelihood == "RobustPoisson":
        return args.method + "_" + args.likelihood + "_" + str(args.gamma) + "gamma"
    elif args.method == "variationalApproxPostHocTrimming":
        return args.method + "_" + str(args.pre_specified_nu) + "nu"
    else:
        assert(args.method in ["variationalApprox", "exact", "oracleLB"])
        return args.method
    
def getPrefix(dataset, args):
    TRAINING_DETAILS_STR = str(args.max_training_itr) + "maxItr" + "_" + str(args.min_training_itr) + "minItr" + "_" + str(args.learning_rate) + "lr"
    METHOD_STR = args.likelihood + "_" + args.covFunc + "_" + get_method_str(args)

    if args.reduced_rank is None:
        reduced_rank_info = ""
    else:
        reduced_rank_info = "_" + str(args.reduced_rank) + "rr" + "_" + str(args.learn_inducing_points) + "inducing_points"

    return dataset + "_" + args.split + "_" + args.noise_type + get_noise_postfix(args.noise_type, args.true_outlier_ratio) + "_" + TRAINING_DETAILS_STR + "_" + METHOD_STR + reduced_rank_info


def saveStatistics(obj, dataset, args, filenameSuffix, folder = ALL_RESULTS_FOLDER):
    filename = folder + getPrefix(dataset, args) + "_" + filenameSuffix
    np.save(filename, obj)
    print("successfully saved to ", filename)
    return

def loadStatistics(dataset, args, filenameSuffix, folder = ALL_RESULTS_FOLDER):
    return loadStatistics_array(dataset, args, filenameSuffix, folder).item()

def loadStatistics_array(dataset, args, filenameSuffix, folder):
    filename = folder + getPrefix(dataset, args) + "_" + filenameSuffix
    # print("load: ", filename)
    return np.load(filename + ".npy", allow_pickle = True)


def assert_valid(a):
    assert(np.all(np.isfinite(a)))
    assert(np.all(np.logical_not(np.isnan(a))))
    return
    


def get_nice_data_name_str(dataset_short_name, kappa):
    
    if dataset_short_name == "Friedman":
        return '''Synthetic ($\kappa=''' + str(kappa) + '''$)'''
    elif dataset_short_name == "asthma":
        return "Asthma"
    elif dataset_short_name == "dengue_iquitos":
        return "Dengue"               
    elif dataset_short_name == "bioChemists":
        return "BioChemists"    
    elif dataset_short_name == "NMES":
        return "NMES"    
    elif dataset_short_name == "bike_sharing_hour":
        return "Bike"
    else:
        assert(False)

def get_nice_noise_type_str(noise_type, short_no_noise = False):
    if noise_type == "noNoise":
        if short_no_noise:
            return "-"
        else:
            return "original"
    elif noise_type == "asymmetric_pos":
        return "random"
    elif noise_type == "max1":
        return "lowest"
    else:
        assert(False)
