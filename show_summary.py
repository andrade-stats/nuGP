import commonSettings
import runExperiments
import copy
import numpy as np
import hyperparam_selection

def show_cmp_all_methods(all_scores_mean, all_scores_std):
    # asssumes lower is better
    min_value = round(np.min(all_scores_mean), 2)

    output_str = ""
    for i in range(all_scores_mean.shape[0]):
        output_str += " & "
        value = round(all_scores_mean[i], 2)
        if value == min_value:
            output_str += '''\\textbf{''' + str(value) + '''} '''
        else:
            output_str += f"{value} "
        output_str += f"({round(all_scores_std[i], 2)})"

    return output_str

def get_nice_outlier_ratio_str(true_outlier_ratio):
    if true_outlier_ratio > 0:
        return " " + str(true_outlier_ratio * 100) + "\\%"
    else:
        return ""

#  & \bfseries Pois & \bfseries $\gamma$-div & \bfseries NB & \bfseries $\nu$-GP & \bfseries $w$-GP & \bfseries Post-Hoc \\
def get_nice_method_str(likelihood, method):
    if likelihood == "Poisson" and method == "variationalApprox":
        return "Pois"
    elif likelihood == "Poisson" and method == "OLRE":
        return "OLRE"
    elif likelihood == "RobustPoisson" and method == "variationalApprox":
        return '''$\\gamma$-div'''
    elif likelihood == "NB" and method == "variationalApprox":
        return "NB"
    elif likelihood == "NB" and method == "trimmedLB_inc":
        return "$\\nu$-GP"
    elif likelihood == "NB" and method == "wGP":
        return "$w$-GP"
    elif likelihood == "NB" and method == "variationalApproxPostHocTrimming":
        return "Post-Hoc"


# run with:
# python show_summary.py --scrps 
# or simply (for NLL):
# python show_summary.py
if __name__ == '__main__':

    original_args = runExperiments.get_standard_settings()
    
    SHOW_NLL = not original_args.scrps

    if SHOW_NLL:
        print("*************** NLL *****************")
    else:
        print("*************** SCRPS *****************")

    # show only results for porposed method
    ALL_METHODS = [("NB", "trimmedLB_inc")]

    # use this for getting full table:
    # ALL_METHODS = [("NB", "trimmedLB_inc"), ("NB", "wGP"), ("RobustPoisson", "variationalApprox"), ("NB", "variationalApproxPostHocTrimming"), ("Poisson", "OLRE"), ("Poisson", "variationalApprox"), ("NB", "variationalApprox")]
    
    previous_dataset = ""

    for dataset_id, (dataset_short_name, kappa, noise_type, true_outlier_ratio) in enumerate(commonSettings.ALL_DATASETS):
        all_nll_mean = np.zeros(len(ALL_METHODS)) * np.nan
        all_nll_std = np.zeros(len(ALL_METHODS)) * np.nan
        all_scrps_mean = np.zeros(len(ALL_METHODS)) * np.nan
        all_scrps_std = np.zeros(len(ALL_METHODS)) * np.nan

        for method_id, (likelihood, method) in enumerate(ALL_METHODS):
            args = copy.deepcopy(original_args)
            args.dataset = dataset_short_name
            args.kappa = kappa
            args.noise_type = noise_type
            args.true_outlier_ratio = true_outlier_ratio
            
            args.likelihood = likelihood
            args.method = method
            
            if dataset_short_name == "bike_sharing_hour" or dataset_short_name == "NMES":
                # --reduced_rank=1000 --learn_inducing_points=True
                args.reduced_rank = 1000
                args.learn_inducing_points = True

            if method == "OLRE":
                args.prior_median = hyperparam_selection.get_best_prior_median(dataset, args)
            elif likelihood == "RobustPoisson":
                args.gamma = hyperparam_selection.get_best_gamma(dataset, args)
            elif method == "variationalApproxPostHocTrimming":
                args.pre_specified_nu=0.2 

            dataset = runExperiments.get_full_dataset_name(args)
            assert(commonSettings.GLOBAL_NUMBER_OF_FOLDS == 10)
        
            score_data = commonSettings.loadStatistics(dataset, args, filenameSuffix = "score_data", folder = "all_summary_data/")

            all_nll_mean[method_id] = np.mean(score_data["all_NLL_mean_collected"])
            all_nll_std[method_id] = np.std(score_data["all_NLL_mean_collected"])

            all_scrps_mean[method_id] = np.mean(score_data["all_scrps_scores"])
            all_scrps_std[method_id] = np.std(score_data["all_scrps_scores"])
            
        
        if previous_dataset != dataset:
            print('''\\midrule''')
            output_str = '''\\multicolumn{7}{c}{\\bfseries ''' +  commonSettings.get_nice_data_name_str(dataset_short_name, kappa) + '''} '''
            output_str += " \\\\"
            output_str += "\n"
            output_str += '''\\midrule''' + "\n"
        else:
            output_str = ""

        if dataset_id == 0:
            output_str += " & "
            output_str += " & ".join(["\\bfseries " + get_nice_method_str(likelihood, method) for (likelihood, method) in ALL_METHODS])
            output_str += " \\\\"
            output_str += "\n"
            output_str += '''\\midrule''' + "\n"
         
        # output_str += '''\\bfseries ''' + commonSettings.get_nice_noise_type_str(noise_type) + get_nice_outlier_ratio_str(true_outlier_ratio)
        output_str += commonSettings.get_nice_noise_type_str(noise_type) + get_nice_outlier_ratio_str(true_outlier_ratio)
        
        if SHOW_NLL:
            output_str += show_cmp_all_methods(all_nll_mean, all_nll_std)
        else:
            output_str += show_cmp_all_methods(all_scrps_mean, all_scrps_std)
        
        output_str += " \\\\"
        
        print(output_str)
        
        previous_dataset = dataset