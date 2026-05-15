import argparse

import numpy as np
import time

import random
import torch

import commonSettings

import evaluation

import commons_GP
import trimmedLB
import gpytorch
import copy

import count_GPs

from commonSettings import ALL_W_PRIOR_FAC
from commonSettings import ALL_PRIOR_MEDIAN

import weightedLikelihoodGP


def load_data(dataset, args):
    return np.load(commonSettings.PREPARED_DATA_FOLDER + dataset + "_" + args.split + "_" + args.noise_type + commonSettings.get_noise_postfix(args.noise_type, args.true_outlier_ratio) + ".npy", allow_pickle = True).item()

# checked
def select_rnd_subset_zeroOne(hold_out_ids_zeroOne, nr_select):
    assert(nr_select >= 10)

    shuffled_ids = np.where(hold_out_ids_zeroOne == 1)[0]
    np.random.shuffle(shuffled_ids)
    
    rnd_ids = shuffled_ids[0:nr_select]
    selected_zeroOne = np.zeros_like(hold_out_ids_zeroOne)
    selected_zeroOne[rnd_ids] = 1
    return selected_zeroOne

# checked
def get_outlier_ids_not_in_tail(all_log_probs, outlier_ids_zeroOne, target_nr_outliers):
    tail_ids = np.argsort(all_log_probs)[0:target_nr_outliers]
    not_in_tails_ids_zeroOne = np.copy(outlier_ids_zeroOne)
    not_in_tails_ids_zeroOne[tail_ids] = 0

    nr_outliers_in_tail = np.sum(outlier_ids_zeroOne[tail_ids])
    return not_in_tails_ids_zeroOne, nr_outliers_in_tail

def run(dataset, args):
    assert(args.true_outlier_ratio >= 0.0 and args.true_outlier_ratio <= 0.4)
    
    commonSettings.setDevice()

    NUMBER_OF_FOLDS = commonSettings.GLOBAL_NUMBER_OF_FOLDS

    np.random.seed(3523421)
    torch.manual_seed(3523421)
    random.seed(3523421)
    
    allData = load_data(dataset, args)

    NR_TRAINING_SAMPLES = allData["all_y_train"][0].shape[0]
    
    print("TOTAL NUMBER OF SAMPLES = ", NR_TRAINING_SAMPLES)
    print("PRE_SPECIFIED_NU = ", args.pre_specified_nu)
    
    startTime = time.time()

    all_NLL = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_NLL_ind = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_NLL_ind_median = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_MSLL = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_RMSE = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_MedianAbsoluteError = np.zeros(NUMBER_OF_FOLDS) * np.nan

    all_runtimes = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_min_losses = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_min_losses_itr = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_last_losses = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_kappa_estimates = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_log_probs_train = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
    all_p_values_left_sided = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
    all_p_values_right_sided = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
    all_abs_residuals_train = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
    
    all_cv_held_out_log_probs = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
    all_cv_held_out_abs_residuals = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
    all_cv_held_out_mean_preds = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
    all_cv_held_out_p_values_left_sided = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
    all_cv_held_out_p_values_right_sided = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan
        

    all_weights = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES)) * np.nan

    if args.method.startswith("trimmedLB"):
        NR_OUTLIER_UPPER_BOUND = int(NR_TRAINING_SAMPLES * args.pre_specified_nu)
        print("NR_OUTLIER_UPPER_BOUND = ", NR_OUTLIER_UPPER_BOUND)
        all_inlier_ids  = np.zeros((NUMBER_OF_FOLDS, NR_TRAINING_SAMPLES - NR_OUTLIER_UPPER_BOUND), dtype = int)
        all_outlier_log_probs = np.zeros((NUMBER_OF_FOLDS, NR_OUTLIER_UPPER_BOUND)) * np.nan
        
    all_RMSE_median_baseline = np.zeros(NUMBER_OF_FOLDS) * np.nan
    all_RMSE_mean_baseline = np.zeros(NUMBER_OF_FOLDS) * np.nan

    allOutlierAUCs = np.zeros(NUMBER_OF_FOLDS) * np.nan
    allOutlierRecalls = np.zeros(NUMBER_OF_FOLDS) * np.nan
    allFirstOutlierQuantile = np.zeros(NUMBER_OF_FOLDS) * np.nan

    all_predictive_dist_at_X = [None] * NUMBER_OF_FOLDS

    for foldId in range(NUMBER_OF_FOLDS):
        
        startTimeOneFold = time.time()

        print(f"********************** data fold id = {foldId} **********************")
        
        X_train = allData["all_X_train"][foldId]
        y_train = allData["all_y_train"][foldId]
        trueOutlierIndicesZeroOne = allData["all_trueOutlierIndicesZeroOne"][foldId]
        
        X_train = commonSettings.getTorchTensor(X_train)
        y_train = commonSettings.getTorchTensor(y_train)
        
        print("X_train.shape = ", X_train.shape)
        print("y_train = ", y_train[0:10])
        print("X_cleanTest.shape = ", allData["all_X_cleanTest"][foldId].shape)
        print("y_cleanTest = ", allData["all_y_cleanTest"][foldId][0:10])
        print("n = ", X_train.shape[0] + allData["all_X_cleanTest"][foldId].shape[0])
        # assert(False)


        all_y_cleanTest_mean = np.mean(allData["all_y_cleanTest"][foldId])
        print("all_y_cleanTest_mean = ", all_y_cleanTest_mean)
        y_train_mean = torch.mean(y_train)
        print("y_train_mean = ", y_train_mean)
        print("y_train_median = ", torch.median(y_train))
        print("dataset = ", dataset)
        
        assert(y_train.shape[0] == NR_TRAINING_SAMPLES)

        if dataset == "housing": 
            # for housing GP is not numerically stable with default setting of jitter 
            JITTER_FOR_GPY_TORCH = 1e-1
        else:
            JITTER_FOR_GPY_TORCH = 1e-6

        with gpytorch.settings.cholesky_jitter(float_value = JITTER_FOR_GPY_TORCH, double_value = JITTER_FOR_GPY_TORCH), gpytorch.settings.cholesky_max_tries(10), gpytorch.settings.variational_cholesky_jitter(float_value = JITTER_FOR_GPY_TORCH, double_value = JITTER_FOR_GPY_TORCH): 
            
            if args.method == "gamma_divergence_CV":
                assert(args.likelihood == "RobustPoisson" and args.gamma > 0)
                all_cv_held_out_log_probs[foldId] = count_GPs.gammaDivergence_CV(args, X_train, y_train, nr_cv_folds = args.nr_folds)
            
            elif args.method == "variationalApprox":
                if args.likelihood == "RobustPoisson":
                    assert(args.gamma > 0)
                else:
                    assert(args.gamma is None)
                gpModel, all_losses = count_GPs.trainCountGP(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = args.likelihood, gamma = args.gamma, useVariationalApprox = True)

            elif args.method == "variationalApproxPostHocTrimming":
                assert(args.pre_specified_nu is not None)
                gpModel, all_losses = count_GPs.trainCountGP(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = args.likelihood, gamma = None, useVariationalApprox = True)

                NR_OUTLIER_UPPER_BOUND = int(NR_TRAINING_SAMPLES * args.pre_specified_nu)
                all_log_probs_train_from_full_model = gpModel.get_all_log_probs_ind(X_train, y_train)  
                nr_inliers = NR_TRAINING_SAMPLES - NR_OUTLIER_UPPER_BOUND
                inlier_ids = np.argsort(-all_log_probs_train_from_full_model)[0:nr_inliers]

                gpModel, all_losses = count_GPs.trainCountGP(args, X_train[inlier_ids], y_train[inlier_ids], covFunc_name = args.covFunc, likelihood_name = args.likelihood, gamma = None, useVariationalApprox = True)

            elif args.method == "exact":
                assert(args.likelihood == "Gaussian" and args.reduced_rank is None)
                gpModel, all_losses = count_GPs.trainCountGP(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = args.likelihood, gamma = None, useVariationalApprox = False)
            elif args.method == "gammaDivergence":
                assert(args.likelihood == "Gaussian") # GPyTorch implementation only supports Gaussian likelihood for gamma-divergence
                gpModel, all_losses = count_GPs.trainCountGP(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = "Gaussian", gamma = args.gamma, useVariationalApprox = True)
            elif args.method == "OLRE":
                gpModel, all_losses, all_weights[foldId] = weightedLikelihoodGP.trainOLRELikelihoodGP_full(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = args.likelihood)
            elif args.method == "wGP":
                gpModel, all_losses, all_weights[foldId] = weightedLikelihoodGP.trainWeightedLikelihoodGP(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = args.likelihood)
            elif args.method == "wGP_trimmed":
                gpModel, all_losses, all_weights[foldId] = weightedLikelihoodGP.trainWeightedLikelihoodGP_trimmed(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = args.likelihood, maxNrOutlierSamples = NR_OUTLIER_UPPER_BOUND)
            elif args.method == "trimmedLB":
                assert(args.pre_specified_nu <= 0.5 and args.pre_specified_nu >= 0.0)
                gpModel, all_losses, all_inlier_ids[foldId] = trimmedLB.trainTrimmedLB(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = args.likelihood, maxNrOutlierSamples = NR_OUTLIER_UPPER_BOUND)
                if args.likelihood == "NB":
                    all_kappa_estimates[foldId] = gpModel.likelihood.kappa.item()
            elif args.method == "trimmedLB_CV":
                assert(args.pre_specified_nu <= 0.5 and args.pre_specified_nu >= 0.0)
                all_cv_held_out_log_probs[foldId], all_cv_held_out_abs_residuals[foldId], all_cv_held_out_mean_preds[foldId], all_cv_held_out_p_values_left_sided[foldId], all_cv_held_out_p_values_right_sided[foldId] = trimmedLB.trainTrimmedLB_CV(args, X_train, y_train, covFunc_name = args.covFunc, likelihood_name = args.likelihood, maxNrOutlierSamples = NR_OUTLIER_UPPER_BOUND, nr_cv_folds = args.nr_folds)
            else:
                assert(False)
            
            if args.method != "trimmedLB_CV" and args.method != "wGP_CV" and args.method != "trimmedLB_CV_adv" and args.method != "gamma_divergence_CV":
                if args.method == "trimmedLB":
                    outlier_ids_zero_one = np.ones(X_train.shape[0])
                    outlier_ids_zero_one[all_inlier_ids[foldId]] = 0
                    all_outlier_log_probs[foldId] =  gpModel.get_all_log_probs_ind(X_train[outlier_ids_zero_one == 1], y_train[outlier_ids_zero_one == 1])
                
                if args.likelihood == "NB":
                    all_p_values_left_sided[foldId], all_p_values_right_sided[foldId] = gpModel.get_one_sided_p_value_NB(X_train, y_train) 

                all_log_probs_train[foldId] = gpModel.get_all_log_probs_ind(X_train, y_train)
                all_abs_residuals_train[foldId], _ = gpModel.getAbsResiduals_and_MeanPredictions(X_train, y_train)

                if args.noise_type != "noNoise":
                    
                    # print("log_probs = ", log_probs_train.shape)
                    # print("mean = ", log_probs_train)
                    # assert(False)
                    # training_nlls = training_nlls.detach().cpu().numpy()
                    # print("training_nlls = ", training_nlls)
                    allOutlierAUCs[foldId], allOutlierRecalls[foldId], allFirstOutlierQuantile[foldId] = evaluation.showOutlierDetectionPerformance_auc_top(trueOutlierIndicesZeroOne, all_log_probs_train[foldId])
                
                if args.split == "trainTestData":
                    
                    X_cleanTest = allData["all_X_cleanTest"][foldId]
                    y_cleanTest = allData["all_y_cleanTest"][foldId]
                    
                    X_cleanTest = commonSettings.getTorchTensor(X_cleanTest)
                    y_cleanTest = commonSettings.getTorchTensor(y_cleanTest)
                    
                    print(f"DEBUG INFO: foldId = {foldId}")
                    print("Start Evaluation")
                    all_NLL[foldId], all_MSLL[foldId], all_RMSE[foldId], all_MedianAbsoluteError[foldId], all_NLL_ind[foldId], all_NLL_ind_median[foldId], all_predictive_dist_at_X[foldId] = gpModel.evaluatePredictions(X_cleanTest, y_cleanTest)

                    # simple median baseline
                    all_RMSE_median_baseline[foldId] = torch.sqrt(torch.mean(torch.square(y_cleanTest - torch.median(y_train))))
                    all_RMSE_mean_baseline[foldId] = torch.sqrt(torch.mean(torch.square(y_cleanTest - torch.mean(y_train))))
                
        
        all_runtimes[foldId] = (time.time() - startTimeOneFold) / 60.0

        if args.method != "trimmedLB_CV" and args.method != "wGP_CV" and args.method != "trimmedLB_CV_adv" and args.method != "gamma_divergence_CV":
            print(f"min loss = {np.nanmin(all_losses)} at iteration = {np.nanargmin(all_losses)}")
            print("********************************************")

            all_min_losses[foldId] = np.nanmin(all_losses)
            all_min_losses_itr[foldId] = np.nanargmin(all_losses)
            
            last_loss_id = all_losses.shape[0] - np.sum(np.isnan(all_losses)) - 1
            all_last_losses[foldId] = all_losses[last_loss_id]

    
    if args.method == "createOutlierData":
        assert(False)
        np.save(commonSettings.PREPARED_DATA_FOLDER + dataset + "_" + args.split + "_" + args.create_noise_type + commonSettings.get_noise_postfix(args.create_noise_type, args.true_outlier_ratio),  allData)
        print(f"finished saving {dataset} with outlier = {args.create_noise_type}")
        return


    if args.method == "trimmedLB_CV" or args.method == "wGP_CV" or args.method == "trimmedLB_CV_adv" or args.method == "gamma_divergence_CV":
        all_training_details = {}
        all_training_details["all_runtimes"] = all_runtimes
        all_training_details["all_cv_held_out_log_probs"] = all_cv_held_out_log_probs
        all_training_details["all_cv_held_out_abs_residuals"] = all_cv_held_out_abs_residuals
        all_training_details["all_cv_held_out_mean_preds"] = all_cv_held_out_mean_preds
        all_training_details["all_cv_held_out_p_values_left_sided"] = all_cv_held_out_p_values_left_sided
        all_training_details["all_cv_held_out_p_values_right_sided"] = all_cv_held_out_p_values_right_sided
        commonSettings.saveStatistics(all_training_details, dataset, args, "all_training_details")
    else:
        all_training_details = {}
        all_training_details["all_runtimes"] = all_runtimes
        all_training_details["all_weights"] = all_weights
        all_training_details["all_min_losses"] = all_min_losses
        all_training_details["all_min_losses_itr"] = all_min_losses_itr
        all_training_details["all_last_losses"] = all_last_losses
        all_training_details["all_log_probs_train"] = all_log_probs_train
        all_training_details["all_p_values_left_sided"] = all_p_values_left_sided
        all_training_details["all_p_values_right_sided"] = all_p_values_right_sided
        all_training_details["all_abs_residuals_train"] = all_abs_residuals_train
        all_training_details["all_kappa_estimates"] = all_kappa_estimates
        # all_training_details["all_mean_cv_log_probs"] = all_mean_cv_log_probs
        # all_training_details["tested_max_outlier_ratios"] = ALL_CV_NU
        commonSettings.saveStatistics(all_training_details, dataset, args, "all_training_details")

        print("******************************")
        print("all_min_losses = ", evaluation.showAvgAndStd_str(all_min_losses))
        print("all_min_losses_itr = ", evaluation.showAvgAndStd_str(all_min_losses_itr))
        print("******************************")
        
        if args.method == "trimmedLB":
            all_trimmedLB_statistics = {}
            all_trimmedLB_statistics["all_inlier_ids"] = all_inlier_ids
            all_trimmedLB_statistics["all_outlier_log_probs"] = all_outlier_log_probs
            commonSettings.saveStatistics(all_trimmedLB_statistics, dataset, args, "all_trimmedLB_statistics")

        if args.split == "trainTestData":

            all_test_data_results = {}
            all_test_data_results["all_NLL"] = all_NLL
            all_test_data_results["all_NLL_ind"] = all_NLL_ind
            all_test_data_results["all_NLL_ind_median"] = all_NLL_ind_median
            all_test_data_results["all_MSLL"] = all_MSLL
            all_test_data_results["all_RMSE"] = all_RMSE
            all_test_data_results["all_MedianAbsoluteError"] = all_MedianAbsoluteError
            all_test_data_results["all_predictive_dist_at_X"] = all_predictive_dist_at_X
            commonSettings.saveStatistics(all_test_data_results, dataset, args, "all_test_data_results")

            print("******************************")
            print("NLL = ", evaluation.showAvgAndStd_str(all_NLL))
            print("NLL (independent, mean) = ", evaluation.showAvgAndStd_str(all_NLL_ind))
            print("NLL (independent, median) = ", evaluation.showAvgAndStd_str(all_NLL_ind_median))
            print("RMSE = ", evaluation.showAvgAndStd_str(all_RMSE))
            print("---")
            print("median baseline RMSE = ", evaluation.showAvgAndStd_str(all_RMSE_median_baseline))
            print("mean baseline RMSE = ", evaluation.showAvgAndStd_str(all_RMSE_mean_baseline))
            print("******************************")

        if args.noise_type != "noNoise":
            all_outlier_results = {}
            all_outlier_results["allOutlierAUCs"] = allOutlierAUCs
            all_outlier_results["allOutlierRecalls"] = allOutlierRecalls
            all_outlier_results["allFirstOutlierQuantile"] = allFirstOutlierQuantile
            commonSettings.saveStatistics(all_outlier_results, dataset, args, "all_outlier_results")
            
            print("******************************")
            print("allOutlierAUCs = ", evaluation.showAvgAndStd_str(allOutlierAUCs))
            print("allOutlierRecalls = ", evaluation.showAvgAndStd_str(allOutlierRecalls))
            print("allFirstOutlierQuantile = ", evaluation.showAvgAndStd_str(allFirstOutlierQuantile))
            print("******************************")

    print("******************************")
    print("FINISHED")
    print("NUMBER_OF_FOLDS = ", NUMBER_OF_FOLDS)
    print("covFunc = ", args.covFunc)
    print("likelihood = ", args.likelihood)
    print("dataset = ", args.dataset)
    print("method = ", args.method)
    if args.method == "gammaDivergence":
        print("gamma = ", args.gamma)
    print("noise_type = ", args.noise_type)
    print("reduced_rank = ", args.reduced_rank)
    print("total runtime (in minutes) = ", (time.time() - startTime) / 60.0)
    return


def get_full_dataset_name(args):
    if args.dataset == "Friedman":
        return f"FriedmanCount_n{args.n}_kappa{args.kappa}"
    else:
        return args.dataset


def get_standard_settings():
    
    # example:
    # python runExperiments.py --likelihood=NB --method=wGP --noise_type=asymmetric_pos --min_training_itr=2000
    # python runExperiments.py --likelihood=NB --method=wGP --noise_type=asymmetric_pos --min_training_itr=10000
    # python runExperiments.py --likelihood=NB --method=wGP --noise_type=noNoise --min_training_itr=10000 --reduced_rank=1000 --dataset=bike_sharing
    # python runExperiments.py --likelihood=NB --method=variationalApprox --noise_type=asymmetric_pos --reduced_rank=1000 --dataset=bike_sharing
    # python runExperiments.py --likelihood=NB --method=variationalApprox --noise_type=asymmetric_pos --reduced_rank=1000 --dataset=bike_sharing
    # python runExperiments.py --likelihood=NB --method=trimmedLB --noise_type=asymmetric_pos --pre_specified_nu=0.2 &&
    # python runExperiments.py --likelihood=NB --method=trimmedLB --noise_type=asymmetric_pos --pre_specified_nu=0.1

    # python runExperiments.py --likelihood=Gaussian --method=gammaDivergence --noise_type=asymmetric_pos
    # python runExperiments.py --likelihood=NB --method=variationalApprox --noise_type=noNoise 
    # python runExperiments.py --likelihood=NB --method=wGP --noise_type=asymmetric_pos --min_training_itr=10000 --wl_prior=three_percent
    # python runExperiments.py --likelihood=NB --method=wGP --noise_type=noNoise --min_training_itr=10000 --wl_prior=three_percent
    # python runExperiments.py --likelihood=NB --method=wGP --noise_type=asymmetric_pos --min_training_itr=10000 --wl_prior=three_percent --dataset=asthma

    parser = argparse.ArgumentParser()
    parser.add_argument("--likelihood", type=str, choices=["NB", "Poisson", "RobustPoisson", "Gaussian", "Student"], default="NB")
    parser.add_argument("--covFunc", type=str, choices=["SE", "Matern"], default="Matern")
    parser.add_argument("--dataset", type=str, default="Friedman") 
    parser.add_argument("--split", type=str, default="trainTestData")
    parser.add_argument("--method", type=str, choices=["variationalApprox", "variationalApproxPostHocTrimming", "trimmedLB", "gammaDivergence",  "wGP", "OLRE"], default="variationalApprox")
    parser.add_argument("--nu_selection_method", type=str, default=None) 
    parser.add_argument("--nu_set", type=str, choices=["coarse", "fine"], default="fine") 
    parser.add_argument("--reduced_rank", type=int, default=None) 
    
    # only used for creating statistics
    parser.add_argument('--fast', type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--scrps', type=bool, default=False, action=argparse.BooleanOptionalAction)

    # only used for synthetic data
    parser.add_argument('--kappa', type=float, default=0.1)
    parser.add_argument('--n', type=int, default=1000)

    # only used for trimmedLB_CV or wGP_CV or trimmedLB_CV_adv
    parser.add_argument("--nr_folds", type=int, default=10)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--step", type=float, default=0.01)

    parser.add_argument('--max_training_itr', type=int, default=commons_GP.L_MAX_STANDARD_NR_TRAINING_ITERATIONS)
    parser.add_argument('--min_training_itr', type=int, default=commons_GP.L_MIN_STANDARD_NR_TRAINING_ITERATIONS)
    parser.add_argument('--learning_rate', type=float, default=commons_GP.L_LEARNING_RATE)

    parser.add_argument('--gamma', type=float, default=None) 
    
    parser.add_argument('--wl_prior_fac', type=float, default=None)
    parser.add_argument('--prior_median', type=float, default=None)

    parser.add_argument("--noise_type", type=str, choices=["noNoise", "random", "lowest"], default="noNoise")
    # parser.add_argument("--noise_type", type=str, choices=["noNoise", "symmetric", "asymmetric_pos", "asymmetric_neg", "focused", "max1"], default="noNoise")
    
    parser.add_argument("--true_outlier_ratio", type=float, default=0.1)
    parser.add_argument("--pre_specified_nu", type=float, default=None)
    parser.add_argument("--learn_inducing_points", type=bool, default=None)
    
    args = parser.parse_args()
    
    if args.noise_type == "random":
        # "asymmetric_pos" corresponds to "random" in the paper
        args.noise_type = "asymmetric_pos"
    elif args.noise_type == "lowest":
        # "max1" corresponds to "lowest" in the paper
        args.noise_type = "max1"
    
    return args



if __name__ == '__main__':

    print("GPyTorch - Version = ", gpytorch.__version__)
    print("PyTorch - Version = ", torch.__version__)
    print("NumPy - Version = ", np.__version__)
    
    args = get_standard_settings()
    dataset_name = get_full_dataset_name(args)

    if args.method == "trimmedLB_CV" or args.method == "trimmedLB":
        assert(args.pre_specified_nu is None)
        assert((args.step == 0.01 and args.start == 0.0) or (args.step == 0.02 and (args.start == 0.0 or args.start == 0.01)))
        # run nu-GP for different nu values, the final nu value will then be selected in create_summary_data.py by calling get_refined_nu from nu_estimation.py
        all_nu_values = np.arange(start = args.start, stop = 0.21, step=args.step)
        for nu in all_nu_values:
            args_cp = copy.deepcopy(args)
            args_cp.pre_specified_nu = nu
            print("**** running with = ", nu)
            run(dataset_name, args_cp)           
    elif args.method == "OLRE":
        assert(args.prior_median is None)
        for med in ALL_PRIOR_MEDIAN:
            args_cp = copy.deepcopy(args)
            args_cp.prior_median = med
            print("**** running with = ", med)
            run(dataset_name, args_cp)
    elif args.method == "wGP":
        assert(args.wl_prior_fac is None)
        for fac in ALL_W_PRIOR_FAC:
            args_cp = copy.deepcopy(args)
            args_cp.wl_prior_fac = fac
            print("**** running with = ", fac)
            run(dataset_name, args_cp)
    elif args.likelihood == "RobustPoisson":
        ALL_GAMMA_VALUES = count_GPs.get_all_gamma_values(args)
        for gamma in ALL_GAMMA_VALUES:
            args_cp = copy.deepcopy(args)
            args_cp.gamma = gamma
            print("**** running with = ", gamma)
            run(dataset_name, args_cp)
    else:
        run(dataset_name, args)


    


