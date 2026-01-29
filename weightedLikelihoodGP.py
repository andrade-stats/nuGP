import torch
import gpytorch
import commons_GP
import count_GPs
import time
import numpy as np
import sklearn.model_selection
import commonSettings

# numerically stable estimate of the inverse softplus
def softplus_inv(c):
    c = torch.tensor([c])
    return c + torch.log(-torch.expm1(-c))

def trainWeightedLikelihoodGP(args, X, y, covFunc_name, likelihood_name):

    n = X.shape[0]
    
    assert(args.wl_prior_fac <= 1000.0 and args.wl_prior_fac >= 0.001)

    EXPECTED_OUTLIER_RATIO = 0.01
    target_mean_inlier = 1.0 - EXPECTED_OUTLIER_RATIO
    prior_alpha = args.wl_prior_fac
    prior_beta = (prior_alpha / target_mean_inlier) - prior_alpha

    print("args.wl_prior_fac = ", args.wl_prior_fac)
    print("prior_alpha = ", prior_alpha)
    print("prior_beta = ", prior_beta)
    

    prior_each_w = torch.distributions.beta.Beta(torch.ones(n) * prior_alpha, torch.ones(n) * prior_beta)

    # if args.wl_prior == "standard":
    #     # P(w < 0.5) = roughly 10%
    #     prior_w_c0 = 0.1
    #     prior_w_c1 = 0.01
    #     prior_each_w = torch.distributions.beta.Beta(torch.ones(n) * prior_w_c0, torch.ones(n) * prior_w_c1)
    # elif args.wl_prior == "three_percent":
    #     # P(w < 0.5) = roughly 3%
    #     prior_w_c0 = 0.1
    #     prior_w_c1 = 0.003
    #     prior_each_w = torch.distributions.beta.Beta(torch.ones(n) * prior_w_c0, torch.ones(n) * prior_w_c1)
    # else:
    #     assert(False)
      
    model, likelihood, mll = count_GPs.get_initial_count_GP(args, X, y, covFunc_name, likelihood_name, gamma = None, useVariationalApprox = True)

    # variational approximation of posterior p(w | y)
    c0 = 1.0 # prior_alpha
    c1 = (c0 / 0.8) - c0 # prior_beta
    qw_c0_unconst = torch.nn.Parameter(torch.ones(n) * softplus_inv(c0))
    qw_c1_unconst = torch.nn.Parameter(torch.ones(n) * softplus_inv(c1))

    # print("qw_c0_unconst = ", softplus_inv(c0))
    # print("softplus(qw_c0_unconst) = ", torch.nn.functional.softplus(softplus_inv(c0)))
    # print("qw_c1_unconst = ", softplus_inv(c1))
    # print("softplus(qw_c1_unconst) = ", torch.nn.functional.softplus(softplus_inv(c1)))
    # print("qw_c1_unconst = ", qw_c1_unconst.mean())
    
    optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()},
                                   {"params": [qw_c0_unconst, qw_c1_unconst]}], lr=args.learning_rate)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    previous_loss = float("inf")
    
    all_losses = np.zeros(args.max_training_itr) * np.nan

    all_expected_weights = None
    startTime = time.time()

    for i in range(args.max_training_itr):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        output = model(X)   # output is the variational distribution q(f)  (i.e. basically the MultivariateNormal returned from the forward of CholeskyVariationalDistribution)

        # print("output = ", output)
        # assert(False)

        expectation_y_train = mll.likelihood.expected_log_prob(y, output)
        assert(len(expectation_y_train.shape) == 1)
        
        qw_c0 = torch.nn.functional.softplus(qw_c0_unconst)
        qw_c1 = torch.nn.functional.softplus(qw_c1_unconst)
        qw = torch.distributions.beta.Beta(qw_c0, qw_c1)
        
        expectation_term = (qw.mean * expectation_y_train).sum()

        kl_divergence_qw = (torch.distributions.kl.kl_divergence(qw, prior_each_w)).sum()
        
        kl_divergence_qf = model.variational_strategy.kl_divergence()
        lower_bound_on_ml = expectation_term - kl_divergence_qf - kl_divergence_qw
        loss = - lower_bound_on_ml / mll.num_data

        # print("qw.mean = ", qw.mean)
        # print("qw_c0 = ", qw_c0)
        # print("qw_c1 = ", qw_c1)
        # print("expectation_term = ", expectation_term)
        # print("kl_divergence_qf = ", kl_divergence_qf)
        # print("kl_divergence_qw = ", kl_divergence_qw)
        # assert(False)

        loss.backward()
        optimizer.step()

        all_losses[i] = loss.detach().cpu().numpy()
        commons_GP.showProgressGP(i, loss, model, likelihood, startTime)

        all_expected_weights = (qw.mean).detach().cpu().numpy()

        if loss >= previous_loss and i >= args.min_training_itr:
            break
        else:
            previous_loss = loss
    
    # assert(i < args.max_training_itr - 1) # if this fails, then this suggests an issue with convergence

    # average_mll_value = mll(model(X), y).item()
    # marginalLikelihood_value = average_mll_value  * y.shape[0]

    learnedGP = commons_GP.BasicGP(model, likelihood)

    return learnedGP, all_losses, all_expected_weights


def trainWeightedLikelihoodGP_CV(args, X, y, covFunc_name, likelihood_name, nr_folds):

    assert(nr_folds is not None)

    kfolds = sklearn.model_selection.KFold(n_splits=nr_folds, random_state=43293, shuffle=True)

    all_log_probs_valid = np.zeros(X.shape[0]) * np.nan
    all_abs_residuals_valid = np.zeros(X.shape[0]) * np.nan

    for i, (train_index, valid_index) in enumerate(kfolds.split(X)):
        gpModel, _, _ = trainWeightedLikelihoodGP(args, X[train_index], y[train_index], covFunc_name, likelihood_name)
        all_log_probs_valid[valid_index] = gpModel.get_all_log_probs_ind(X[valid_index], y[valid_index])  
        all_abs_residuals_valid[valid_index] = gpModel.getAbsResiduals(X[valid_index], y[valid_index]) 

    assert(np.all(~np.isnan(all_log_probs_valid)))
    return all_log_probs_valid, all_abs_residuals_valid


def trainWeightedLikelihoodGP_trimmed(args, X, y, covFunc_name, likelihood_name, maxNrOutlierSamples):

    n = X.shape[0]
    
    assert(args.wl_prior_fac <= 300.0 and args.wl_prior_fac >= 0.1)

    EXPECTED_INLIER_RATIO = 0.03
    target_mean_inlier = 1.0 - EXPECTED_INLIER_RATIO
    prior_alpha = 0.1 * args.wl_prior_fac
    prior_beta = (prior_alpha / target_mean_inlier) - prior_alpha

    prior_each_w = torch.distributions.beta.Beta(torch.ones(maxNrOutlierSamples) * prior_alpha, torch.ones(maxNrOutlierSamples) * prior_beta)
      
    model, likelihood, mll = count_GPs.get_initial_count_GP(args, X, y, covFunc_name, likelihood_name, gamma = None, useVariationalApprox = True)

    # variational approximation of posterior p(w | y)
    c0 = prior_alpha
    c1 = prior_beta
    qw_c0_unconst = torch.nn.Parameter(torch.ones(n) * softplus_inv(c0))
    qw_c1_unconst = torch.nn.Parameter(torch.ones(n) * softplus_inv(c1))

    optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()},
                                   {"params": [qw_c0_unconst, qw_c1_unconst]}], lr=args.learning_rate)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    previous_loss = float("inf")
    
    all_losses = np.zeros(args.max_training_itr) * np.nan

    all_expected_weights = None
    startTime = time.time()

    for i in range(args.max_training_itr):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        output = model(X)   # output is the variational distribution q(f)  (i.e. basically the MultivariateNormal returned from the forward of CholeskyVariationalDistribution)

        expectation_y_train = mll.likelihood.expected_log_prob(y, output)
        assert(len(expectation_y_train.shape) == 1)

        expectation_y_train, sorted_indices = torch.sort(expectation_y_train)
        expectation_y_train_inliers = expectation_y_train[maxNrOutlierSamples:mll.num_data]
        
        # potential outliers
        expectation_y_train_outliers = expectation_y_train[0:maxNrOutlierSamples]

        outlier_ids = sorted_indices[0:maxNrOutlierSamples]
        qw_c0 = torch.nn.functional.softplus(qw_c0_unconst[outlier_ids])
        qw_c1 = torch.nn.functional.softplus(qw_c1_unconst[outlier_ids])
        qw = torch.distributions.beta.Beta(qw_c0, qw_c1)
        
        expectation_term = expectation_y_train_inliers.sum()
        expectation_term += (qw.mean * expectation_y_train_outliers).sum()

        kl_divergence_qw = (torch.distributions.kl.kl_divergence(qw, prior_each_w)).sum()
        
        kl_divergence_qf = model.variational_strategy.kl_divergence()
        lower_bound_on_ml = expectation_term - kl_divergence_qf - kl_divergence_qw
        loss = - lower_bound_on_ml / mll.num_data

        # print("expectation_term = ", expectation_term)
        # print("kl_divergence_qf = ", kl_divergence_qf)
        # print("kl_divergence_qw = ", kl_divergence_qw)
        
        loss.backward()
        optimizer.step()

        all_losses[i] = loss.detach().cpu().numpy()
        commons_GP.showProgressGP(i, loss, model, likelihood, startTime)

        outlier_ids = outlier_ids.detach().cpu().numpy()
        all_expected_weights = np.ones(n)
        all_expected_weights[outlier_ids] = (qw.mean).detach().cpu().numpy()

        if loss >= previous_loss and i >= args.min_training_itr:
            break
        else:
            previous_loss = loss
    
    assert(i < args.max_training_itr - 1) # if this fails, then this suggests an issue with convergence

    # average_mll_value = mll(model(X), y).item()
    # marginalLikelihood_value = average_mll_value  * y.shape[0]

    learnedGP = commons_GP.BasicGP(model, likelihood)

    return learnedGP, all_losses, all_expected_weights



# implementation of the observation-level random effect (OLRE) model
# see e.g.: "Using observation-level random effects to model overdispersion in count data in ecology and evolution", Harrison XA (2014) . PeerJ
def trainOLRELikelihoodGP_full(args, X, y, covFunc_name, likelihood_name):

    # Full OLRE: integrate over q(z) instead of plug-in E[z]
    # y_i | f_i, z_i ~ Poisson(exp(f_i) + z_i), z_i ~ N(0, sigma_r^2)
    # Variational family: q(f) (from GP) and independent q(z_i)=N(mu_i, sigma_i^2)

    assert(likelihood_name == "Poisson"), "OLRE is defined for Poisson likelihood in the paper."

    n = X.shape[0]

    model, likelihood, mll = count_GPs.get_initial_count_GP(
        args, X, y, covFunc_name, likelihood_name, gamma=None, useVariationalApprox=True
    )

    # median of Exponential prior on sigma_r 
    # "Model-Based Smoothing with Integrated Wiener Processes and Overlapping Splines" recommends 0.1, but this might be problem dependent. 
    PRIOR_MEDIAN = args.prior_median 

    # Variational parameters for q(z)
    qz_mu = torch.nn.Parameter(torch.zeros(n))
    INIT_SIGMA = PRIOR_MEDIAN
    qz_sigma_unconst = torch.nn.Parameter(torch.ones(n) * softplus_inv(INIT_SIGMA))


    # Prior std parameter sigma_r (positive)
    INIT_SIGMA_R = PRIOR_MEDIAN
    sigma_r_unconst = torch.nn.Parameter(softplus_inv(INIT_SIGMA_R))

    print("X.dtype = ", X.device)
    print("sigma_r_unconst.dtype = ", sigma_r_unconst.device)
    # assert(False)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
            {"params": [qz_mu, qz_sigma_unconst, sigma_r_unconst]},
        ],
        lr=args.learning_rate,
    )

    model.train()
    likelihood.train()

    previous_loss = float("inf")
    all_losses = np.zeros(args.max_training_itr) * np.nan

    quadrature = gpytorch.utils.quadrature.GaussHermiteQuadrature1D().to(commonSettings.DEVICE)
    startTime = time.time()

    EPS = torch.tensor(1.0e-8)
    NR_MCMC_SAMPLES_INNER_LOOP = 100  # MC samples for z expectation

    for i in range(args.max_training_itr):
        optimizer.zero_grad()

        # q(f)
        function_dist = model(X)

        # q(z): params
        qz_sigma = torch.nn.functional.softplus(qz_sigma_unconst)
        sigma_r = torch.nn.functional.softplus(sigma_r_unconst)

        # Define E_{q(z)}[ log p(y | f, z) ] as a function of f-samples for outer quadrature
        def expected_log_prob_over_z(function_samples):
            # function_samples: [n_f, n], where n_f is the number of monte carlo samples and n is the number of training data points
            n_f = function_samples.shape[0]

            # MC sample S_Z draws from q(z)
            eps = torch.randn(NR_MCMC_SAMPLES_INNER_LOOP, n)
            z_samples = qz_mu.view(1, -1) + eps * qz_sigma.view(1, -1)  # [S_Z, n]

            # Broadcast to combine with f samples: result [n_f, S_Z, n]

            f_term = function_samples.unsqueeze(1)
            z_term = z_samples.unsqueeze(0)

            # print("function_samples = ", function_samples.shape)
            # print("f_term = ", f_term.shape)
            # print("z_samples = ", z_samples.shape)
            # print("z_term = ", z_term.shape)

            # add noise terms for each training data points to f_term (before taking exp)
            rates = torch.exp(f_term + z_term)

            # print("rates = ", rates.shape)
            
            rates = torch.clamp(rates, min=EPS)  # ensure valid Poisson rate

            y_b = y.view(1, 1, -1).expand(n_f, NR_MCMC_SAMPLES_INNER_LOOP, -1)
            pois = torch.distributions.poisson.Poisson(rate=rates)
            logp = pois.log_prob(y_b)  # [n_f, S_Z, n]

            # Average over z samples → [n_f, n]
            return logp.mean(dim=1)

        # E_{q(f)} [ E_{q(z)} log p(y | f,z) ]
        expectation_y_train = quadrature(expected_log_prob_over_z, function_dist)
        assert(len(expectation_y_train.shape) == 1)

        expectation_term = expectation_y_train.sum()

        # KL terms
        kl_divergence_qf = model.variational_strategy.kl_divergence()
        qz = torch.distributions.Normal(loc=qz_mu, scale=qz_sigma)
        pz = torch.distributions.Normal(loc=torch.zeros_like(qz_mu), scale=torch.ones_like(qz_sigma) * sigma_r)
        kl_divergence_qz = torch.distributions.kl.kl_divergence(qz, pz).sum()

        # Exponential prior on sigma_r 
        # For Exp(rate), median = ln(2)/rate 
        exp_rate = torch.tensor(np.log(2.0) / PRIOR_MEDIAN)
        log_prior_sigma_r = torch.log(exp_rate) - exp_rate * sigma_r
        # Scale by num_data so the prior has dataset-size-invariant strength under our per‑datapoint loss
        prior_term = mll.num_data * log_prior_sigma_r

        lower_bound_on_ml = expectation_term - kl_divergence_qf - kl_divergence_qz + prior_term
        loss = -lower_bound_on_ml / mll.num_data

        loss.backward()
        optimizer.step()

        all_losses[i] = loss.detach().cpu().numpy()
        commons_GP.showProgressGP(i, loss, model, likelihood, startTime)

        if loss >= previous_loss and i >= args.min_training_itr:
            break
        else:
            previous_loss = loss

    learnedGP = commons_GP.BasicGP(model, likelihood)
    all_mu = qz_mu.detach().cpu().numpy()
    return learnedGP, all_losses, all_mu
