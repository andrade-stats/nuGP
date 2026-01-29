import commonSettings
import runExperiments
import copy
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':

    ALL_METHODS = []
    
    original_args = runExperiments.get_standard_settings()
    original_args.likelihood = "NB"
    original_args.method = "trimmedLB_inc"
    ALL_METHODS.append(original_args)

    original_args = runExperiments.get_standard_settings()
    original_args.likelihood = "NB"
    original_args.method = "wGP"
    original_args.wl_prior_fac = 1.0
    ALL_METHODS.append(original_args)

    original_args = runExperiments.get_standard_settings()
    original_args.likelihood = "NB"
    original_args.method = "wGP"
    original_args.wl_prior_fac = 10.0
    ALL_METHODS.append(original_args)
    
    all_nll_results = []
    all_scrps_results = []
    
    previous_dataset = ""
    for (dataset_short_name, kappa, noise_type, true_outlier_ratio) in commonSettings.ALL_DATASETS:
        all_outlier_estimates_mean = np.zeros(len(ALL_METHODS)) * np.nan
        all_outlier_estimates_std = np.zeros(len(ALL_METHODS)) * np.nan

        for method_id, original_args in enumerate(ALL_METHODS):
            args = copy.deepcopy(original_args)
            args.dataset = dataset_short_name
            args.kappa = kappa
            args.noise_type = noise_type
            args.true_outlier_ratio = true_outlier_ratio
            
            if dataset_short_name == "bike_sharing_hour" or dataset_short_name == "NMES":
                # --reduced_rank=1000 --learn_inducing_points=True
                args.reduced_rank = 1000
                args.learn_inducing_points = True

            dataset = runExperiments.get_full_dataset_name(args)
            
            assert(commonSettings.GLOBAL_NUMBER_OF_FOLDS == 10)
            
            if args.method == "trimmedLB_inc":
                all_outlier_ratio_estimates = commonSettings.loadStatistics_array(dataset, args, filenameSuffix = "outlier_ratio_estimates", folder = "all_summary_data/")
            elif args.method == "wGP":
                all_training_details = commonSettings.loadStatistics(dataset, args, "all_training_details")
                all_outlier_ratio_estimates = np.mean(all_training_details["all_weights"] < 0.5, axis = 1)
            else:
                assert(False)
            
            all_outlier_estimates_mean[method_id] = np.mean(all_outlier_ratio_estimates)
            all_outlier_estimates_std[method_id] = np.std(all_outlier_ratio_estimates)
            
        bestId = np.argmin(np.abs(all_outlier_estimates_mean - true_outlier_ratio))

        true_outlier_ratio = str(true_outlier_ratio * 100) + "\%"

        if dataset_short_name != "Friedman":
            true_outlier_ratio = '''$\geq$ ''' + true_outlier_ratio

        if previous_dataset != dataset:
            print('''\\midrule''')
            output_str = '''\\multicolumn{5}{c}{\\bfseries ''' +  commonSettings.get_nice_data_name_str(dataset_short_name, kappa) + '''} '''
            output_str += " \\\\"
            output_str += "\n"
            output_str += '''\\midrule''' + "\n"
        else:
            output_str = ""

        output_str += commonSettings.get_nice_noise_type_str(noise_type, short_no_noise = True)
        output_str += " & "
        output_str += true_outlier_ratio
        
        for i in range(len(ALL_METHODS)):
            output_str += " & "
            if i == bestId:
                output_str += '''\\textbf{''' + str(round(all_outlier_estimates_mean[i] * 100, 2)) + '''\%} '''
            else:
                output_str += f"{round(all_outlier_estimates_mean[i] * 100, 2)}\% "
            output_str += f"({round(all_outlier_estimates_std[i] * 100, 2)}\%)"

        # output_str += " & ".join(all_outlier_estimates)
        output_str += " \\\\"

        print(output_str)
        
        previous_dataset = dataset
