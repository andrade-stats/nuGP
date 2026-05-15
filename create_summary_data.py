import commonSettings
import runExperiments
import evaluation
import hyperparam_selection
import numpy as np
import copy

# python create_summary_data.py --noise_type=noNoise &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.1  &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.1  &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.05  &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.05  &&
# python create_summary_data.py --kappa=0.5 --noise_type=noNoise &&
# python create_summary_data.py --kappa=0.5 --noise_type=lowest --true_outlier_ratio=0.05  &&
# python create_summary_data.py --kappa=0.5 --noise_type=random --true_outlier_ratio=0.05  &&
# python create_summary_data.py --kappa=0.5 --noise_type=lowest --true_outlier_ratio=0.1 &&
# python create_summary_data.py --kappa=0.5 --noise_type=random --true_outlier_ratio=0.1 &&
# python create_summary_data.py --noise_type=noNoise --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.1 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.1 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.05 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.05 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=noNoise --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.05 --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.05 --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.1 --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.1 --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True &&
# python create_summary_data.py --dataset=asthma --noise_type=noNoise &&
# python create_summary_data.py --dataset=asthma --noise_type=lowest --true_outlier_ratio=0.1  &&
# python create_summary_data.py --dataset=asthma --noise_type=random --true_outlier_ratio=0.1  &&
# python create_summary_data.py --dataset=asthma --noise_type=lowest --true_outlier_ratio=0.05  &&
# python create_summary_data.py --dataset=asthma --noise_type=random --true_outlier_ratio=0.05  &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=noNoise &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=lowest --true_outlier_ratio=0.1  &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=random --true_outlier_ratio=0.1  &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=lowest --true_outlier_ratio=0.05  &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=random --true_outlier_ratio=0.05  &&
# python create_summary_data.py --dataset=bioChemists --noise_type=noNoise &&
# python create_summary_data.py --dataset=bioChemists --noise_type=lowest --true_outlier_ratio=0.1  &&
# python create_summary_data.py --dataset=bioChemists --noise_type=random --true_outlier_ratio=0.1  &&
# python create_summary_data.py --dataset=bioChemists --noise_type=lowest --true_outlier_ratio=0.05  &&
# python create_summary_data.py --dataset=bioChemists --noise_type=random --true_outlier_ratio=0.05

# SKIPS SCRPS SCORE CALCULATION:
# python create_summary_data.py --noise_type=noNoise --fast &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.1  --fast &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.1  --fast &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --kappa=0.5 --noise_type=noNoise --fast &&
# python create_summary_data.py --kappa=0.5 --noise_type=lowest --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --kappa=0.5 --noise_type=random --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --kappa=0.5 --noise_type=lowest --true_outlier_ratio=0.1 --fast &&
# python create_summary_data.py --kappa=0.5 --noise_type=random --true_outlier_ratio=0.1 --fast &&
# python create_summary_data.py --noise_type=noNoise --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.1 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.1 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.05 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.05 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=noNoise --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.05 --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.05 --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.1 --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --noise_type=random --true_outlier_ratio=0.1 --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True --fast &&
# python create_summary_data.py --dataset=asthma --noise_type=noNoise --fast &&
# python create_summary_data.py --dataset=asthma --noise_type=lowest --true_outlier_ratio=0.1  --fast &&
# python create_summary_data.py --dataset=asthma --noise_type=random --true_outlier_ratio=0.1  --fast &&
# python create_summary_data.py --dataset=asthma --noise_type=lowest --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --dataset=asthma --noise_type=random --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=noNoise --fast &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=lowest --true_outlier_ratio=0.1  --fast &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=random --true_outlier_ratio=0.1  --fast &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=lowest --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --dataset=dengue_iquitos --noise_type=random --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --dataset=bioChemists --noise_type=noNoise --fast &&
# python create_summary_data.py --dataset=bioChemists --noise_type=lowest --true_outlier_ratio=0.1  --fast &&
# python create_summary_data.py --dataset=bioChemists --noise_type=random --true_outlier_ratio=0.1  --fast &&
# python create_summary_data.py --dataset=bioChemists --noise_type=lowest --true_outlier_ratio=0.05  --fast &&
# python create_summary_data.py --dataset=bioChemists --noise_type=random --true_outlier_ratio=0.05  --fast

if __name__ == '__main__':

    original_args = runExperiments.get_standard_settings()
    
    # show only results for porposed method
    ALL_SETTINGS = [("NB", "trimmedLB_inc")]

    # use this for getting full table:
    # ALL_SETTINGS = [("Poisson", "variationalApprox"), ("Poisson", "OLRE"), ("RobustPoisson", "variationalApprox"), ("NB", "variationalApprox"), ("NB", "trimmedLB_inc"), ("NB", "wGP"), ("NB", "variationalApproxPostHocTrimming")]
    
    all_nll_results = []
    all_scrps_results = []
    
    all_refined_nu = None

    for (likelihood, method) in ALL_SETTINGS:
        args = copy.deepcopy(original_args)
        args.likelihood = likelihood
        args.method = method
        
        dataset = runExperiments.get_full_dataset_name(args)
        allData = runExperiments.load_data(dataset, args)

        assert(commonSettings.GLOBAL_NUMBER_OF_FOLDS == 10)
        
        all_predictive_dist_collected = commonSettings.GLOBAL_NUMBER_OF_FOLDS  * [None]
        all_NLL_mean_collected = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan

        if method == "trimmedLB_CV" or method == "trimmedLB_inc":
            all_refined_nu = hyperparam_selection.trimmed_LB_nu_selection(dataset, args)
            
            for foldId in range(commonSettings.GLOBAL_NUMBER_OF_FOLDS):        
                args_cp = copy.deepcopy(args)
                args_cp.pre_specified_nu = all_refined_nu[foldId]
                args_cp.method = "trimmedLB"
                all_test_data_results = commonSettings.loadStatistics(dataset, args_cp, "all_test_data_results")
                all_predictive_dist_collected[foldId] = all_test_data_results["all_predictive_dist_at_X"][foldId]
                all_NLL_mean_collected[foldId] = all_test_data_results['all_NLL_ind'][foldId]

        elif method == "wGP":
            ALPHA_0 = 1.0

            for foldId in range(commonSettings.GLOBAL_NUMBER_OF_FOLDS):        
                args_cp = copy.deepcopy(args)
                args_cp.wl_prior_fac = ALPHA_0
                all_test_data_results = commonSettings.loadStatistics(dataset, args_cp, "all_test_data_results")
                all_predictive_dist_collected[foldId] = all_test_data_results["all_predictive_dist_at_X"][foldId]
                all_NLL_mean_collected[foldId] = all_test_data_results['all_NLL_ind'][foldId]
                
        elif method == "gamma_divergence_CV":
            
            all_best_gamma_values = hyperparam_selection.get_best_gamma(dataset, args)
            
            for foldId in range(commonSettings.GLOBAL_NUMBER_OF_FOLDS):        
                args_cp = copy.deepcopy(args)
                args_cp.gamma = all_best_gamma_values[foldId]
                args_cp.likelihood = "RobustPoisson"
                args_cp.method = "variationalApprox"
                all_test_data_results = commonSettings.loadStatistics(dataset, args_cp, "all_test_data_results")
                all_predictive_dist_collected[foldId] = all_test_data_results["all_predictive_dist_at_X"][foldId]
                all_NLL_mean_collected[foldId] = all_test_data_results['all_NLL_ind'][foldId]
                
        else:

            if method == "OLRE":
                args.prior_median = hyperparam_selection.get_best_prior_median(dataset, args)
                print("**** prior_median = ", args.prior_median)
            elif likelihood == "RobustPoisson":
                args.gamma = hyperparam_selection.get_best_gamma(dataset, args)
                print("**** gamma = ", args.gamma)
            elif method == "variationalApproxPostHocTrimming":
                args.pre_specified_nu=0.2 

            all_test_data_results = commonSettings.loadStatistics(dataset, args, "all_test_data_results")
            all_predictive_dist_collected = all_test_data_results["all_predictive_dist_at_X"]
            all_NLL_mean_collected = all_test_data_results['all_NLL_ind']


        assert(len(all_predictive_dist_collected) == commonSettings.GLOBAL_NUMBER_OF_FOLDS)

        all_scrps_scores = np.zeros(commonSettings.GLOBAL_NUMBER_OF_FOLDS) * np.nan
        
        if args.fast:
            print("SKIP MARGINAL CALIBRATION DIAGRAM AND SCRPS")
        else:
            empirical_cdf, predictive_cdf = evaluation.get_marginal_calibration_diagram_data(allData, all_predictive_dist_collected)
            plot_data = {}
            plot_data["empirical_cdf"] = empirical_cdf
            plot_data["predictive_cdf"] = predictive_cdf
            commonSettings.saveStatistics(plot_data, dataset, args, filenameSuffix = "plot_data", folder = "all_summary_data/")

            for foldId in range(commonSettings.GLOBAL_NUMBER_OF_FOLDS):
                all_scrps_scores[foldId] = evaluation.get_scaled_crps(allData, all_predictive_dist_collected, foldId)

        assert(all_scrps_scores.shape[0] == 10)
        assert(all_NLL_mean_collected.shape[0] == 10)

        score_data = {}
        score_data["all_scrps_scores"] = all_scrps_scores
        score_data["all_NLL_mean_collected"] = all_NLL_mean_collected
        commonSettings.saveStatistics(score_data, dataset, args, filenameSuffix = "score_data", folder = "all_summary_data/")

        all_nll_results.append(evaluation.showAvgAndStd_str(all_NLL_mean_collected))
        all_scrps_results.append(evaluation.showAvgAndStd_str(all_scrps_scores))
        

    print("all_refined_nu = ", all_refined_nu)
    print(f"***** {dataset} -  {args.noise_type} *********")
    print("  nll & " + " & ".join(all_nll_results) + " \\\\")
    print("  scrps & " + " & ".join(all_scrps_results) + " \\\\")
