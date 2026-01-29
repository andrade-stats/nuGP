import commonSettings
import gpytorch
import copy

import torch
import commons_GP
import count_GPs
import time
import numpy as np

import sklearn.model_selection


def trainTrimmedLB(args, X, y, covFunc_name, likelihood_name, maxNrOutlierSamples):

    model, likelihood, mll = count_GPs.get_initial_count_GP(args, X, y, covFunc_name, likelihood_name, gamma = None, useVariationalApprox = True)

    optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=args.learning_rate) # need to include likelihood explicitly
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    previous_loss = float("inf")
    
    all_losses = np.zeros(args.max_training_itr) * np.nan
    all_inlier_ids = None


    startTime = time.time()

    for i in range(args.max_training_itr):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        output = model(X)   # output is the variational distribution q(f)  (i.e. basically the MultivariateNormal returned from the forward of CholeskyVariationalDistribution)
    
        expectation_y_train = mll.likelihood.expected_log_prob(y, output)
        
        expectation_y_train, sorted_indices = torch.sort(expectation_y_train)
        expectation_y_train_trimmed = expectation_y_train[maxNrOutlierSamples:mll.num_data]
        all_inlier_ids = sorted_indices[maxNrOutlierSamples:mll.num_data]
        
        expectation_term = expectation_y_train_trimmed.sum(-1)
        
        kl_divergence_qf = model.variational_strategy.kl_divergence()
        lower_bound_on_ml = (expectation_term - kl_divergence_qf) 
        loss = - lower_bound_on_ml / mll.num_data

        loss.backward()
        optimizer.step()

        all_losses[i] = loss.detach().cpu().numpy()
        commons_GP.showProgressGP(i, loss, model, likelihood, startTime)

        if loss >= previous_loss and i >= args.min_training_itr:
            break
        else:
            previous_loss = loss
    
    assert(i < args.max_training_itr - 1) # if this fails, then this suggests an issue with convergence

    # average_mll_value = mll(model(X), y).item()
    # marginalLikelihood_value = average_mll_value  * y.shape[0]

    learnedGP = commons_GP.BasicGP(model, likelihood)

    return learnedGP, all_losses, all_inlier_ids.detach().cpu().numpy()



def trainTrimmedLB_CV(args, X, y, covFunc_name, likelihood_name, maxNrOutlierSamples, nr_cv_folds):

    assert(nr_cv_folds is not None)

    if args.cv_seed_id == 1:
        RANDOM_STATE_SEED = 43293
    else:
        assert(args.cv_seed_id == 2)
        RANDOM_STATE_SEED = 984

    kfolds = sklearn.model_selection.KFold(n_splits=nr_cv_folds, random_state=RANDOM_STATE_SEED, shuffle=True)

    all_log_probs_valid = np.zeros(X.shape[0]) * np.nan
    all_p_values_left_sided = np.zeros(X.shape[0]) * np.nan
    all_p_values_right_sided = np.zeros(X.shape[0]) * np.nan
    all_abs_residuals_valid = np.zeros(X.shape[0]) * np.nan
    all_mean_pred_valid = np.zeros(X.shape[0]) * np.nan

    for i, (train_index, valid_index) in enumerate(kfolds.split(X)):
        gpModel, _, _ = trainTrimmedLB(args, X[train_index], y[train_index], covFunc_name, likelihood_name, maxNrOutlierSamples)
        all_log_probs_valid[valid_index] = gpModel.get_all_log_probs_ind(X[valid_index], y[valid_index])  
        all_p_values_left_sided[valid_index], all_p_values_right_sided[valid_index] = gpModel.get_one_sided_p_value_NB(X[valid_index], y[valid_index])  
        all_abs_residuals_valid[valid_index], all_mean_pred_valid[valid_index] = gpModel.getAbsResiduals_and_MeanPredictions(X[valid_index], y[valid_index]) 

    assert(np.all(~np.isnan(all_log_probs_valid)))
    assert(np.all(~np.isnan(all_p_values_left_sided)))
    assert(np.all(~np.isnan(all_p_values_right_sided)))
    return all_log_probs_valid, all_abs_residuals_valid, all_mean_pred_valid, all_p_values_left_sided, all_p_values_right_sided



def get_potential_outlier_inlier_ids(all_log_probs, nu):
    assert(nu == 0.2)

    n = all_log_probs.shape[0]
    max_outlier = int(n * nu)
    
    sorted_log_prob_ids = np.argsort(all_log_probs)
    outlierIds = sorted_log_prob_ids[0:max_outlier]
    inlierIds = sorted_log_prob_ids[max_outlier:n]

    return outlierIds, inlierIds


def trainTrimmedLB_CV_adv(args, dataset, foldId, nu_for_cv_inlier, X, y, covFunc_name, likelihood_name, maxNrOutlierSamples, nr_cv_folds):

    assert(nr_cv_folds is not None)

    args_trimmedLB = copy.deepcopy(args)
    args_trimmedLB.method = "trimmedLB"
    args_trimmedLB.pre_specified_nu = nu_for_cv_inlier
    all_training_details_trimmedLB = commonSettings.loadStatistics(dataset, args_trimmedLB, "all_training_details")
    outlier_ids, inlier_ids = get_potential_outlier_inlier_ids(all_training_details_trimmedLB["all_log_probs_train"][foldId], nu = args_trimmedLB.pre_specified_nu)

    X_outlier = X[outlier_ids]
    y_outlier = y[outlier_ids]

    X_inlier = X[inlier_ids]
    y_inlier = y[inlier_ids]

    kfolds = sklearn.model_selection.KFold(n_splits=nr_cv_folds, random_state=43293, shuffle=True)

    all_log_probs_valid = np.zeros(X_inlier.shape[0]) * np.nan
    all_abs_residuals_valid = np.zeros(X_inlier.shape[0]) * np.nan

    for i, (train_index, valid_index) in enumerate(kfolds.split(X_inlier)):

        X_train = torch.vstack((X_inlier[train_index], X_outlier))
        y_train = torch.cat((y_inlier[train_index], y_outlier))
        
        gpModel, _, _ = trainTrimmedLB(args, X_train, y_train, covFunc_name, likelihood_name, maxNrOutlierSamples)
        all_log_probs_valid[valid_index] = gpModel.get_all_log_probs_ind(X_inlier[valid_index], y_inlier[valid_index])  
        all_abs_residuals_valid[valid_index] = gpModel.getAbsResiduals(X_inlier[valid_index], y_inlier[valid_index]) 

    assert(np.all(~np.isnan(all_log_probs_valid)))
    return all_log_probs_valid, all_abs_residuals_valid