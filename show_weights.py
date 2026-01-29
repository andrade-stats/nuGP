import commonSettings
import runExperiments
import copy
import numpy as np
import matplotlib.pyplot as plt

LEGEND_FONT_SIZE = 15
BIG_SIZE = 20
AXIS_FONT_SIZE = 10
MEDIUM_SIZE = 15

def get_full_name(dataset_short_name, kappa):
    if dataset_short_name == "Friedman":
        return dataset_short_name + "_" + str(kappa)
    else:
        return dataset_short_name

    
def plot_sorted_weights(w_setting_no_noise, w_setting_with_noise, dataset_short_name, kappa):
    assert(w_setting_no_noise.shape[0] == w_setting_with_noise.shape[0])

    w_setting1_sorted = np.sort(w_setting_no_noise)
    w_setting2_sorted = np.sort(w_setting_with_noise)

    # show only lowest 10% weights
    MAX_LENGTH = int(w_setting_no_noise.shape[0] * 0.1)
    w_setting1_sorted = w_setting1_sorted[0:MAX_LENGTH]
    w_setting2_sorted = w_setting2_sorted[0:MAX_LENGTH]

    x = np.arange(w_setting1_sorted.shape[0])
    fig, ax = plt.subplots()
    ax.plot(x, w_setting1_sorted, label='no outliers')
    ax.plot(x, w_setting2_sorted, label='with outliers')

    # Set axis labels and title with desired font size
    ax.set_xlabel('lower quantiles', fontsize=MEDIUM_SIZE)
    ax.set_ylabel(r'$\mathbb{E}_{q_{\boldsymbol{\alpha}, \boldsymbol{\beta}}}[w_i]$', fontsize=MEDIUM_SIZE)
    ax.set_title(commonSettings.get_nice_data_name_str(dataset_short_name, kappa), fontsize=MEDIUM_SIZE)
    
    # Set x-ticks to show 0%..10% across the plotted segment
    L = w_setting1_sorted.shape[0]
    ax.set_xticks(np.linspace(0, L - 1, 6))
    ax.set_xticklabels(["0%", "2%", "4%", "6%", "8%", "10%"])
    
    # Add legend for the two curves
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid()

    fig.savefig("all_plots/" + "w_" + get_full_name(dataset_short_name, kappa) + ".pdf")
    # plt.show()


if __name__ == '__main__':

    ALL_METHODS = []
    
    original_args = runExperiments.get_standard_settings()
    original_args.likelihood = "NB"
    original_args.method = "wGP"
    original_args.wl_prior_fac = 1.0
    ALL_METHODS.append(original_args)

    all_datasets_w_no_noise = {}
    all_datasets_w_with_noise = {}


    
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
            assert(args.method == "wGP")

            all_training_details = commonSettings.loadStatistics(dataset, args, "all_training_details")
            # all_outlier_ratio_estimates = np.mean(all_training_details["all_weights"] < 0.5, axis = 1)

            if (true_outlier_ratio == 0.0 or true_outlier_ratio == 0.05) and (noise_type == "asymmetric_pos" or noise_type == "noNoise"):
                w_one_fold = all_training_details["all_weights"][0]
                
                if true_outlier_ratio == 0.0:
                    all_datasets_w_no_noise[get_full_name(dataset_short_name, kappa)] = w_one_fold
                else:
                    all_datasets_w_with_noise[get_full_name(dataset_short_name, kappa)] = w_one_fold
                

    for dataset_short_name_full in all_datasets_w_no_noise.keys():
        if dataset_short_name_full.startswith("Friedman"):
            dataset_short_name = dataset_short_name_full.split("_")[0]
            kappa = float(dataset_short_name_full.split("_")[1])
        else:
            dataset_short_name = dataset_short_name_full
            kappa = None
        
        plot_sorted_weights(all_datasets_w_no_noise[dataset_short_name_full], all_datasets_w_with_noise[dataset_short_name_full], dataset_short_name, kappa)
