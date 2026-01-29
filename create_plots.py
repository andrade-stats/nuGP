import commonSettings
import runExperiments
import numpy as np
import copy

import matplotlib.pyplot as plt

LEGEND_FONT_SIZE = 15
BIG_SIZE = 20
AXIS_FONT_SIZE = 10
MEDIUM_SIZE = 15

# python showSummary.py --noise_type=noNoise 
# python showSummary.py --dataset=asthma --noise_type=noNoise 
# python showSummary.py --dataset=dengue_iquitos --noise_type=noNoise 
# python showSummary.py --dataset=bioChemists --noise_type=noNoise 
# python showSummary.py --dataset=bioChemists --noise_type=asymmetric_pos 
# python showSummary.py --reduced_rank=1000 --dataset=bike_sharing_hour --learn_inducing_points=True  --noise_type=noNoise 

# python showSummary.py --dataset=asthma --noise_type=asymmetric_pos 
# python showSummary.py --dataset=dengue_iquitos --noise_type=asymmetric_pos 
# python showSummary.py --reduced_rank=1000 --dataset=bike_sharing --learn_inducing_points=True  --noise_type=asymmetric_pos 

# python showSummary.py --noise_type=asymmetric_pos
# python showSummary.py --dataset=asthma --noise_type=noNoise
# python showSummary.py --dataset=asthma --noise_type=noNoise

# python showSummary.py --dataset=bike_sharing_day --noise_type=noNoise 
# python showSummary.py --dataset=bike_sharing_day --noise_type=asymmetric_pos 

# python showSummary.py --dataset=bioChemists --noise_type=noNoise 
# python showSummary.py --dataset=bioChemists --noise_type=asymmetric_pos 
# python showSummary.py --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True  --noise_type=asymmetric_pos
# python showSummary.py --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True  --noise_type=noNoise
# python showSummary.py --dataset=NMES --reduced_rank=1000 --learn_inducing_points=True  --noise_type=noNoise
# python showSummary.py --noise_type=max1 --true_outlier_ratio=0.05 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True

# python create_plots.py --noise_type=noNoise --kappa=0.5

# python create_plots.py --noise_type=noNoise --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True
# python create_plots.py --noise_type=max1 --true_outlier_ratio=0.05 --dataset=bike_sharing_hour --reduced_rank=1000 --learn_inducing_points=True


def get_nice_method_str(likelihood, method):
    if likelihood == "Poisson" and method == "variationalApprox":
        return "Pois"
    elif likelihood == "NB" and method == "variationalApprox":
        return "NB"
    elif likelihood == "NB" and method == "trimmedLB_inc":
        return r"$\nu$-GP" # "nu-GP"
    else:
        assert(False)


if __name__ == '__main__':

    ALL_DATA_SET_PAIRS = [(("Friedman", 0.1, "noNoise", 0), ("Friedman", 0.1, "asymmetric_pos", 0.05))]
    ALL_DATA_SET_PAIRS += [(("Friedman", 0.5, "noNoise", 0), ("Friedman", 0.5, "asymmetric_pos", 0.05))]    
    ALL_DATA_SET_PAIRS += [(("asthma", None, "noNoise", 0), ("asthma", None, "asymmetric_pos", 0.05))]
    ALL_DATA_SET_PAIRS += [(("dengue_iquitos", None, "noNoise", 0), ("dengue_iquitos", None, "asymmetric_pos", 0.05))]
    ALL_DATA_SET_PAIRS += [(("bioChemists", None, "noNoise", 0), ("bioChemists", None, "asymmetric_pos", 0.05))]
    ALL_DATA_SET_PAIRS += [(("NMES", None, "noNoise", 0), ("NMES", None, "asymmetric_pos", 0.05))]
    ALL_DATA_SET_PAIRS += [(("bike_sharing_hour", None, "noNoise", 0), ("bike_sharing_hour", None, "asymmetric_pos", 0.05))]
    
    for pair_id, SELECTED_DATASET_PAIR in enumerate(ALL_DATA_SET_PAIRS):
        
        original_args = runExperiments.get_standard_settings()

        ALL_SETTINGS = [("Poisson", "variationalApprox"), ("NB", "variationalApprox"), ("NB", "trimmedLB_inc")]

        data_name_title = commonSettings.get_nice_data_name_str(SELECTED_DATASET_PAIR[0][0], SELECTED_DATASET_PAIR[0][1])

        figure_size = (6, 8)

        fig, all_sub_plots = plt.subplots(2, figsize=figure_size) 
        
        fig.suptitle(data_name_title, fontsize=BIG_SIZE)

        for subplot_id in range(2):
            (dataset_short_name, kappa, noise_type, true_outlier_ratio) = SELECTED_DATASET_PAIR[subplot_id]

            args_dataset = copy.deepcopy(original_args)
            args_dataset.dataset = dataset_short_name
            args_dataset.kappa = kappa
            args_dataset.noise_type = noise_type
            args_dataset.true_outlier_ratio = true_outlier_ratio

            if dataset_short_name == "bike_sharing_hour" or dataset_short_name == "NMES":
                # --reduced_rank=1000 --learn_inducing_points=True
                args_dataset.reduced_rank = 1000
                args_dataset.learn_inducing_points = True

            current_sub_plot = all_sub_plots[subplot_id]

            current_sub_plot.grid(axis='x', color='0.95')
            current_sub_plot.grid(axis='y', color='0.95')

            plt_handles = []
            
            all_y_values = None

            # Compute Q1 and Q3 of y (test targets),
            dataset_name_for_data = runExperiments.get_full_dataset_name(args_dataset)
            allData_for_quartiles = runExperiments.load_data(dataset_name_for_data, args_dataset)
            all_y_cleanTest_concat = np.concatenate(allData_for_quartiles["all_y_cleanTest"], axis=0)
            q1 = np.quantile(all_y_cleanTest_concat, 0.25) 
            q3 = np.quantile(all_y_cleanTest_concat, 0.75) 
            
            for i, (likelihood, method) in enumerate(ALL_SETTINGS):
                args = copy.deepcopy(args_dataset)
                args.likelihood = likelihood
                args.method = method
                
                dataset = runExperiments.get_full_dataset_name(args)
                allData = runExperiments.load_data(dataset, args)

                assert(commonSettings.GLOBAL_NUMBER_OF_FOLDS == 10)
                
                plot_data = commonSettings.loadStatistics(dataset, args, filenameSuffix = "plot_data", folder = "all_summary_data/")

                all_diff = plot_data["predictive_cdf"] - plot_data["empirical_cdf"]
                mean_diff = np.mean(all_diff, axis = 0)
                std = np.std(all_diff, axis = 0)

                max_value = mean_diff.shape[0]
                all_y_values = np.arange(max_value)
                
                plt_handles += current_sub_plot.plot(all_y_values, mean_diff, label = get_nice_method_str(likelihood, method))  #  color = colors[i])
                current_sub_plot.fill_between(all_y_values, mean_diff - std, mean_diff + std, alpha=0.1) # facecolor='yellow', 
                
                print("plt_handles = ", plt_handles)


            if subplot_id == 0:
                current_sub_plot.set_title("Original", fontsize = MEDIUM_SIZE)
                if pair_id == 1 or pair_id == 6:
                    current_sub_plot.legend(handles=plt_handles, fontsize = LEGEND_FONT_SIZE)
            else:
                current_sub_plot.set_title("With Additional Outliers (random 5%)", fontsize = MEDIUM_SIZE)
                current_sub_plot.set_xlabel("y", size = MEDIUM_SIZE)
            

            # subplot.set_ylim(y_limits)

            # for tick in subplot.xaxis.get_major_ticks():
            #     tick.label.set_fontsize(fontsize=STANDARD_FONT_SIZE) 
            
            # for tick in subplot.yaxis.get_major_ticks():
            #     tick.label.set_fontsize(fontsize=STANDARD_FONT_SIZE) 

            current_sub_plot.xaxis.set_tick_params(labelsize=AXIS_FONT_SIZE)
            current_sub_plot.yaxis.set_tick_params(labelsize=AXIS_FONT_SIZE)

            if pair_id == 0 or pair_id == 2 or pair_id == 4 or pair_id == 6:
                current_sub_plot.set_ylabel("CDF difference", fontsize = MEDIUM_SIZE)
            
            current_sub_plot.axhline(0.0, linestyle='--', color = "black")

            # Draw Q1 and Q3 as vertical lines 
            current_sub_plot.axvline(q1, linestyle=':', color='gray', alpha=0.8)
            current_sub_plot.axvline(q3, linestyle=':', color='gray', alpha=0.8)

        # Align lower subplot x-axis with upper subplot
        all_sub_plots[1].set_xbound(all_sub_plots[0].get_xbound())
        

        plt.tight_layout() # pad = 2.5)
        filename = SELECTED_DATASET_PAIR[0][0]
        if SELECTED_DATASET_PAIR[0][1] is not None:
             filename += str(SELECTED_DATASET_PAIR[0][1])
        plt.savefig("all_plots/" + filename + ".pdf")
        # plt.show()
        print("data_name_title = ", data_name_title)
