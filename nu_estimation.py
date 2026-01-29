import commonSettings
import numpy as np
import copy
import negative_binomial_helper
from commonSettings import ALL_PRE_SPECFIFIED_NU
from commonSettings import TAU
import matplotlib.pyplot as plt

def get_inlier_ids_after_trimming(inlier_scores, nr_inlier):
    return np.argsort(-inlier_scores)[0:nr_inlier]
    

def get_inliers_based_on_p_values(y_true_all, y_mean_preds_all, cv_held_out_log_probs_all, nr_inliers):

    # ************** 1. find best kappa (and S) under constraint |S| = nr_inliers ****************
    kappa = 1.0
    inlier_ids = get_inlier_ids_after_trimming(cv_held_out_log_probs_all, nr_inliers)

    nr_iter = 0
    while(True):
        nr_iter += 1
        kappa, best_nll= negative_binomial_helper.get_best_kappa_nll(y_true_all[inlier_ids], y_mean_preds_all[inlier_ids], initial_kappa = kappa)

        _, pmf = negative_binomial_helper.get_NB_p_values(y_true_all, y_mean_preds_all, kappa)
        new_inlier_ids = get_inlier_ids_after_trimming(pmf, nr_inliers)

        assert(new_inlier_ids.shape[0] == inlier_ids.shape[0])
        intersection = np.intersect1d(new_inlier_ids, inlier_ids)
        nr_differences = new_inlier_ids.shape[0] - intersection.shape[0]
        if nr_differences == 0:
            break
        
        inlier_ids = np.copy(new_inlier_ids)

    # ************** 2. identify inliers based on p-values **************

    p_values, _ = negative_binomial_helper.get_NB_p_values(y_true_all, y_mean_preds_all, kappa)

    print("nr of inliers (before) = ", inlier_ids.shape[0])
    new_nr_inliers = np.sum(p_values >= TAU)
    print("new_nr_inliers = ", new_nr_inliers)
    return new_nr_inliers, best_nll



def get_outlier_stats_trimmedLB(dataset, args_orig, nu):
    args = copy.deepcopy(args_orig)
    args.likelihood = "NB"
    args.method = "trimmedLB"
    args.pre_specified_nu = nu
    all_outlier_results = commonSettings.loadStatistics(dataset, args, "all_outlier_results")
    return all_outlier_results

def get_training_stats_trimmedLB(dataset, args_orig, nu):
    args = copy.deepcopy(args_orig)
    args.likelihood = "NB"
    args.method = "trimmedLB"
    args.pre_specified_nu = nu
    all_training_details = commonSettings.loadStatistics(dataset, args, "all_training_details")
    return all_training_details


def get_training_stats_cv(dataset, args_orig, nu):
    assert(args_orig.method == "trimmedLB_CV")

    args = copy.deepcopy(args_orig)
    args.pre_specified_nu = nu
    all_training_details = commonSettings.loadStatistics(dataset, args, "all_training_details")

    all_cv_held_out_log_probs_all_folds = all_training_details["all_cv_held_out_log_probs"]
    all_cv_held_out_mean_preds_all_folds = all_training_details["all_cv_held_out_mean_preds"]

    if "all_cv_held_out_p_values" in all_training_details:
        all_cv_held_out_p_values_all_folds = all_training_details["all_cv_held_out_p_values"]
    else:
        all_cv_held_out_p_values_all_folds = all_training_details["all_cv_held_out_p_values_right_sided"]

    return all_cv_held_out_log_probs_all_folds, all_cv_held_out_mean_preds_all_folds, all_cv_held_out_p_values_all_folds


def analyze_p_values(dataset, args):

    NU = 0.08
    _, _, all_cv_held_out_p_values_all_folds = get_training_stats(dataset, args, nu = NU)

    # all_outlier_results = get_outlier_stats_trimmedLB(dataset, args, nu = NU)
    all_training_details = get_training_stats_trimmedLB(dataset, args, nu = NU)
    
    foldId = 2

    # print("allOutlierRecalls = ", all_outlier_results["allOutlierRecalls"])
    # print("AUC = ", all_outlier_results["allOutlierAUCs"][foldId])
    # print("outlier recall = ", all_outlier_results["allOutlierRecalls"][foldId])
    
    all_p_values = all_training_details["all_p_values_right_sided"][foldId]
    # all_p_values = all_cv_held_out_p_values_all_folds[foldId]

    n = all_p_values.shape[0]
    max_outlier = int(n * 0.2)

    print("n = ", n)
    new_outlier_ratio = np.sum(all_p_values < TAU) / n

    print("nr samples p-values < 0.01 = ", np.sum(all_p_values < TAU))
    print("new_outlier_ratio = ", new_outlier_ratio)
    # print("true_outlier_ratio = ", args.true_outlier_ratio)

    # print("max_outlier = ", max_outlier)
    all_criteria_sorted = -np.sort(-all_p_values)[(n-max_outlier):n]
    
    # print("all_criteria_sorted = ", all_criteria_sorted)

    x = np.arange(all_criteria_sorted.shape[0])
    fig, ax = plt.subplots()
    ax.plot(x, all_criteria_sorted)

    ax.set(xlabel='sorted ids', ylabel='value',
        title='Values')
    # ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid()

    # fig.savefig("test.png")
    plt.show()
    assert(False)


def get_all_p_values(dataset, args, nu):
    if args.method == "trimmedLB_CV":
        _, _, all_cv_held_out_p_values_all_folds = get_training_stats_cv(dataset, args, nu)
        return all_cv_held_out_p_values_all_folds
    elif args.method == "trimmedLB_inc":
        all_training_details = get_training_stats_trimmedLB(dataset, args, nu)
        return all_training_details["all_p_values_right_sided"]
    else:
        assert(False)

# implements Algorithm 3 to get improved nu estimate
def get_refined_nu(dataset, args, foldId):
    ALL_PRE_SPECFIFIED_NU_SORTED = -np.sort(-ALL_PRE_SPECFIFIED_NU)
    MOST_CONSERVATIVE_NU = ALL_PRE_SPECFIFIED_NU_SORTED[0]
    
    # print("ALL_PRE_SPECFIFIED_NU_SORTED = ", ALL_PRE_SPECFIFIED_NU_SORTED)
    # print("MOST_CONSERVATIVE_NU = ", MOST_CONSERVATIVE_NU)
    # assert(False)

    all_p_values = get_all_p_values(dataset, args, nu = MOST_CONSERVATIVE_NU)
    
    # analyze_p_values(dataset, args)

    assert(args.likelihood == "NB")
    n = all_p_values[foldId].shape[0]
    
    previous_conservative_nu = MOST_CONSERVATIVE_NU
    print("previous_conservative_nu = ", previous_conservative_nu)

    while(True):
        assert(n == all_p_values[foldId].shape[0])

        new_nr_inliers = np.sum(all_p_values[foldId] >= TAU)

        new_outlier_ratio = (1.0 - (new_nr_inliers / n))
        new_nu_ratio_id = np.sum(ALL_PRE_SPECFIFIED_NU_SORTED >= new_outlier_ratio) - 1
        new_conservative_nu = ALL_PRE_SPECFIFIED_NU_SORTED[new_nu_ratio_id]
        
        print("new_outlier_ratio = ", new_outlier_ratio)
        print("new_conservative_nu = ", new_conservative_nu)

        if new_conservative_nu >= previous_conservative_nu:
            final_conservative_nu = new_conservative_nu
            outlier_ratio_estimate = new_outlier_ratio
            break
        
        all_p_values = get_all_p_values(dataset, args, nu = new_conservative_nu)
        previous_conservative_nu = new_conservative_nu
        

    print(f"final nu = {final_conservative_nu}")
    
    return final_conservative_nu, outlier_ratio_estimate


