from typing import Any
import gpytorch.likelihoods
import gpytorch.constraints
import torch
from torch import Tensor
from torch.distributions import NegativeBinomial
from torch.distributions import Poisson

import commons_GP
from commons_GP import VariationalGPModel
from commons_GP import ExactGPModel
from commons_GP import CONST_MEAN_FUNCTION
from commons_GP import getCovFunc
import commonSettings

import gpytorch.utils.quadrature
import gpytorch.distributions

import negative_binomial_helper
import numpy as np

import abc
import sklearn.model_selection
import scipy

INITIAL_KAPPA = 1.0


class PoissonLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([])
    ) -> None:
        super().__init__()

    def forward(self, latent_f: Tensor, *args: Any, **kwargs: Any) -> Poisson:
        return torch.distributions.poisson.Poisson(rate = torch.exp(latent_f))


# for some datasets too high-values are numerically instable, see also
# Chapter 2.7 of "Minimum Gamma Divergence for Regression and Classification Problems", 2025
def get_all_gamma_values(args):
    if args.dataset == "bike_sharing":
        ALL_GAMMA_VALUES = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    elif args.dataset == "bike_sharing_day" or args.dataset == "bike_sharing_hour":
        ALL_GAMMA_VALUES = [0.0001, 0.00001, 0.000001, 0.0000001]
    elif args.dataset == "dengue_iquitos":
        ALL_GAMMA_VALUES = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    elif args.dataset == "NMES":
        ALL_GAMMA_VALUES = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    else:
        ALL_GAMMA_VALUES = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

    return ALL_GAMMA_VALUES

# checked
class GammaDivergencePoissonLikelihood(gpytorch.likelihoods.Likelihood, abc.ABC):


    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([]),
        gamma = 1.03
    ) -> None:
        super().__init__()
        self.quadrature = gpytorch.utils.quadrature.GaussHermiteQuadrature1D()
        self.gamma = gamma
        
        # to improve numerical stability
        self.EPSILON = commonSettings.getTorchTensor(0.00001)

    def expected_log_prob(
        self, observations: Tensor, function_dist: gpytorch.distributions.MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        
        def neg_gamma_divergence(function_samples):
            poisson = self.forward(function_samples, *args, **kwargs)
            nr_mc_samples = poisson.mean.shape[0]
            nr_obs = observations.shape[0]

            observations_broadcasted = torch.broadcast_to(observations, (nr_mc_samples, nr_obs))
            
            gamma_ratio = - self.gamma / (self.gamma + 1.0)
            
            first_term = (self.gamma * observations_broadcasted) * torch.log(poisson.mean)
            second_term = gamma_ratio * (poisson.mean ** (self.gamma + 1))

            result =  (1.0 / self.gamma) * torch.exp(first_term + second_term)
            
            # non_finite_pos = torch.logical_not(torch.isfinite(result))
            # print("first_term = ", first_term[non_finite_pos])
            # print("second_term = ", second_term[non_finite_pos])
            assert(torch.all(torch.logical_not(torch.isnan(result))))
            assert(torch.all(torch.isfinite(result)))
            return result
        
        log_prob = self.quadrature(neg_gamma_divergence, function_dist)
        return log_prob
    
    def forward(self, latent_f: Tensor, *args: Any, **kwargs: Any) -> Poisson:
        rates = torch.exp(latent_f) + self.EPSILON
        # illegal_values = torch.where(rates <= 0)
        # rates += self.EPSILON
        # illegal_values = rates[rates <= 0]
        # if illegal_values.shape[0] > 0:
        # print("illegal_values = ", illegal_values)
        # print("checked1 = ", torch.any(rates <= 0.0))
        # print("checked2 = ", torch.all(rates > 0.0))
        # print("checked3 = ", torch.all(rates >= 0.0))
        # print("check4 = ", torch.any(torch.isnan(rates)))
        assert(torch.all(rates > 0))
        return torch.distributions.poisson.Poisson(rate = rates)

    def log_marginal(
        self, observations: Tensor, function_dist: gpytorch.distributions.MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        assert(False)


class NegativeBinomialLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):

    def __init__(
        self,
        batch_shape: torch.Size = torch.Size([]),
        fix_kappa = False
    ) -> None:
        super().__init__()

        kappa_constraint = gpytorch.constraints.Positive()
        
        self.raw_kappa = torch.nn.Parameter(torch.zeros(*batch_shape, 1))
        self.register_constraint("raw_kappa", kappa_constraint)

        if fix_kappa:
            self.raw_kappa.requires_grad = False
        

    @property
    def kappa(self) -> Tensor:
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value: Tensor) -> None:
        self._set_kappa(value)

    def _set_kappa(self, value: Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kappa)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))

    def forward(self, latent_f: Tensor, *args: Any, **kwargs: Any) -> NegativeBinomial:
        logits = latent_f + torch.log(self.kappa)
        if torch.isnan(logits).any():
            print("WARNING: FOUND NAN VALUE:")
            print("logits shape = ", logits.shape)
            print("logits = ")
            print(logits)
            logits[torch.isnan(logits)] = 0.0

        return torch.distributions.negative_binomial.NegativeBinomial(total_count = 1.0 / self.kappa, logits = logits)
    






def get_initial_count_GP(args, X, y, covFunc_name, likelihood_name, gamma, useVariationalApprox):

    if (likelihood_name == "Gaussian") and (gamma is None) and (not useVariationalApprox):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X, y, likelihood, meanFunc = CONST_MEAN_FUNCTION, covFunc = getCovFunc(covFunc_name, X.shape[1]))
        likelihood.noise = commons_GP.INITIAL_SIGMA_SQUARE

        model = commonSettings.setDataType(model)
        likelihood = commonSettings.setDataType(likelihood)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        print("USE EXACT GP")

    else:
        assert(useVariationalApprox)
        # initialize likelihood and model
        model = VariationalGPModel(X, meanFunc = CONST_MEAN_FUNCTION, covFunc = getCovFunc(covFunc_name, X.shape[1]), reducedRank = args.reduced_rank, learn_inducing_points = args.learn_inducing_points)

        if likelihood_name == "NB":
            likelihood = NegativeBinomialLikelihood()
            likelihood.kappa = INITIAL_KAPPA
        elif likelihood_name == "NB_fixed_kappa":
            likelihood = NegativeBinomialLikelihood(fix_kappa=True)
            likelihood.kappa = args.kappa
        elif likelihood_name == "Poisson":
            likelihood = PoissonLikelihood()
        elif likelihood_name == "RobustPoisson":
            likelihood = GammaDivergencePoissonLikelihood(gamma = gamma)
        elif likelihood_name == "Student":
            likelihood = gpytorch.likelihoods.StudentTLikelihood()
            likelihood.noise = commons_GP.INITIAL_SIGMA_SQUARE
        elif likelihood_name == "Gaussian":
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            likelihood.noise = commons_GP.INITIAL_SIGMA_SQUARE
        else:
            assert(False)
        
        model = commonSettings.setDataType(model)
        likelihood = commonSettings.setDataType(likelihood)

        if (gamma is None) or (likelihood_name == "RobustPoisson"):
            print("USE VARIATIONAL APPROXIMATION")
            # "Loss" for GPs - the marginal log likelihood
            # num_data refers to the number of training datapoints
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data = y.shape[0])
        else:
            print("USE GAMMA DIVERGENCE")
            # note that gamma(GammaRobustVariationalELBO) = 1 + gamma("Variational Inference based on Robust Divergences") 
            mll = gpytorch.mlls.GammaRobustVariationalELBO(likelihood, model, num_data=y.shape[0], gamma = gamma + 1.0)

    return model, likelihood, mll


# checked
def trainCountGP(args, X, y, covFunc_name, likelihood_name, gamma, useVariationalApprox):

    model, likelihood, mll = get_initial_count_GP(args, X, y, covFunc_name, likelihood_name, gamma, useVariationalApprox)

    model, likelihood, _, _, all_losses = commons_GP.standardTraining(args, model, likelihood, mll, X, y)
    
    learnedGP = commons_GP.BasicGP(model, likelihood)

    return learnedGP, all_losses


def gammaDivergence_CV(args, X, y, nr_cv_folds):
    assert(args.likelihood == "RobustPoisson" and args.gamma > 0)
    assert(nr_cv_folds is not None)

    RANDOM_STATE_SEED = 43293
    kfolds = sklearn.model_selection.KFold(n_splits=nr_cv_folds, random_state=RANDOM_STATE_SEED, shuffle=True)

    all_log_probs_valid = np.zeros(X.shape[0]) * np.nan

    for i, (train_index, valid_index) in enumerate(kfolds.split(X)):

        gpModel, _ = trainCountGP(args, X[train_index], y[train_index], covFunc_name = args.covFunc, likelihood_name = args.likelihood, gamma = args.gamma, useVariationalApprox = True)
        all_log_probs_valid[valid_index] = gpModel.get_all_log_probs_ind(X[valid_index], y[valid_index])  
        
    assert(np.all(~np.isnan(all_log_probs_valid)))
    return all_log_probs_valid


def get_inliers_based_on_NB_p_values(gp_model, X, y_true, tau):
    
    predictive_distribution_at_X = commons_GP.getPredictions(gp_model.model, gp_model.likelihood, X)
    meanPredictions = commons_GP.getMeanPredictions(predictive_distribution_at_X)
    y_mean_preds = meanPredictions.detach().cpu().numpy()
    
    kappa = gp_model.likelihood.kappa.item()
    
    y_true_numpy = y_true.detach().cpu().numpy()
    p_values = negative_binomial_helper.get_NB_p_values(y_true_numpy, y_mean_preds, kappa)
    print("p_values = ", p_values)
    print("p_values.shape = ", p_values.shape)

    print("tau = ", tau)

    X_inliers = X[p_values >= tau]
    y_inliers = y_true[p_values >= tau]
    return X_inliers, y_inliers