import commonSettings
import evaluation
import numpy as np
import copy
import count_GPs

from commonSettings import ALL_PRE_SPECFIFIED_NU
from commonSettings import ALL_W_PRIOR_FAC

import nu_estimation


def trimmed_LB_nu_selection(dataset, args):
    assert(commonSettings.GLOBAL_NUMBER_OF_FOLDS == 10)

    all_NLL_ind_median_collected = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan
    all_NLL_ind_mean_collected = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan
    all_MedianAbsoluteError_collected = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan
    all_refined_nu = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan
    all_outlier_ratio_estimates = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan

    for current_fold_id in range(commonSettings.GLOBAL_NUMBER_OF_FOLDS):
        refined_nu, all_outlier_ratio_estimates[current_fold_id] = nu_estimation.get_refined_nu(dataset, args, foldId = current_fold_id)
        
        print("refined_nu = ", refined_nu)
        args_cp = copy.deepcopy(args)
        args_cp.pre_specified_nu = refined_nu
        args_cp.method = "trimmedLB"
        all_test_data_results = commonSettings.loadStatistics(dataset, args_cp, "all_test_data_results")
            
        assert(all_test_data_results['all_NLL_ind_median'].shape[0] == commonSettings.GLOBAL_NUMBER_OF_FOLDS)
        all_NLL_ind_median_collected[current_fold_id] = all_test_data_results['all_NLL_ind_median'][current_fold_id]
        all_NLL_ind_mean_collected[current_fold_id] = all_test_data_results['all_NLL_ind'][current_fold_id]
        all_MedianAbsoluteError_collected[current_fold_id] = all_test_data_results['all_MedianAbsoluteError'][current_fold_id]
        all_refined_nu[current_fold_id] = refined_nu
        
    
    print(f"---------------- {dataset} ({args.noise_type}) --------------------")
    print(f"method = {args.method}, likelihood = {args.likelihood}")
    print("all_refined_nu = ", all_refined_nu)
    print("all_outlier_ratio_estimates = ", all_outlier_ratio_estimates)
    print(f"Median NLL = {evaluation.showAvgAndStd_str(all_NLL_ind_median_collected)}, Mean NLL = {evaluation.showAvgAndStd_str(all_NLL_ind_mean_collected)}")
    print(f"Median Absolute Error = {evaluation.showAvgAndStd_str(all_MedianAbsoluteError_collected)}")

    commonSettings.saveStatistics(all_refined_nu, dataset, args, "all_refined_nu", folder = "all_results_hyper_params/")
    commonSettings.saveStatistics(all_outlier_ratio_estimates, dataset, args, "outlier_ratio_estimates", folder = "all_summary_data/")    
    return all_refined_nu


def get_least_informative_prior_subject_to_nu_constraint(dataset, args_orig):

    MOST_CONSERVATIVE_NU = np.max(ALL_PRE_SPECFIFIED_NU)
    ALL_W_PRIOR_FAC_SORTED = np.sort(ALL_W_PRIOR_FAC)

    print("MOST_CONSERVATIVE_NU = ", MOST_CONSERVATIVE_NU)
    print("ALL_W_PRIOR_FAC_SORTED = ", ALL_W_PRIOR_FAC_SORTED)

    all_best_prior_fac = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan
    all_outlier_ratio_estimates = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan

    for current_prior_fac in ALL_W_PRIOR_FAC_SORTED:
        args = copy.deepcopy(args_orig)
        args.wl_prior_fac = current_prior_fac
        all_training_details = commonSettings.loadStatistics(dataset, args, "all_training_details")

        all_estimated_outlier_ratios_this_fac = 1.0 - np.mean(all_training_details["all_weights"], axis = 1)
        print("all_estimated_outlier_ratios_this_fac = ", all_estimated_outlier_ratios_this_fac)

        select_condition = np.logical_and((all_estimated_outlier_ratios_this_fac < MOST_CONSERVATIVE_NU), np.isnan(all_best_prior_fac))
        all_outlier_ratio_estimates[select_condition] = all_estimated_outlier_ratios_this_fac[select_condition]
        all_best_prior_fac[select_condition] = current_prior_fac
        
    print("all_best_prior_fac = ", all_best_prior_fac)
    assert(np.all(np.logical_not(np.isnan(all_best_prior_fac))))

    print("all_outlier_ratio_estimates = ", all_outlier_ratio_estimates)
    commonSettings.saveStatistics(all_outlier_ratio_estimates, dataset, args, "outlier_ratio_estimates", folder = "all_summary_data/")    
    
    return all_best_prior_fac



def get_best_gamma(dataset, args):
    ALL_GAMMA_VALUES = count_GPs.get_all_gamma_values(args)
        
    all_scores = np.zeros(len(ALL_GAMMA_VALUES))
    for gamma_id, gamma in enumerate(ALL_GAMMA_VALUES):
        args_cp = copy.deepcopy(args)
        args_cp.gamma = gamma

        # oracle selection of gamma-value
        all_test_data_results = commonSettings.loadStatistics(dataset, args_cp, "all_test_data_results")
        all_scores[gamma_id] = np.mean(all_test_data_results['all_NLL_ind'])
        
    best_id = np.argmin(all_scores)
    return ALL_GAMMA_VALUES[best_id]


def get_best_prior_median(dataset, args):
        
    all_scores = np.zeros(len(commonSettings.ALL_PRIOR_MEDIAN))
    for med_id, med in enumerate(commonSettings.ALL_PRIOR_MEDIAN):
        args_cp = copy.deepcopy(args)
        args_cp.prior_median = med

        # oracle selection of prior median
        all_test_data_results = commonSettings.loadStatistics(dataset, args_cp, "all_test_data_results")
        all_scores[med_id] = np.mean(all_test_data_results['all_NLL_ind'])
        
    best_id = np.argmin(all_scores)
    return commonSettings.ALL_PRIOR_MEDIAN[best_id]


def get_best_gamma_cv(dataset, args):
    ALL_GAMMA_VALUES = count_GPs.get_all_gamma_values(args)
        
    all_best_gamma_values = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan

    for foldId in range(commonSettings.GLOBAL_NUMBER_OF_FOLDS):  

        all_scores = np.zeros(len(ALL_GAMMA_VALUES))
        for gamma_id, gamma in enumerate(ALL_GAMMA_VALUES):
            args_cp = copy.deepcopy(args)
            args_cp.gamma = gamma

            all_training_details = commonSettings.loadStatistics(dataset, args_cp, "all_training_details")
            all_scores[gamma_id] = np.median(all_training_details["all_cv_held_out_log_probs"])

        best_id = np.argmax(all_scores)
        all_best_gamma_values[foldId] = ALL_GAMMA_VALUES[best_id]
    
    print("all_best_gamma_values = ", all_best_gamma_values)
    return all_best_gamma_values

