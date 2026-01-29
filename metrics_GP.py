import torch
import gpytorch
import math

def negative_log_predictive_density(pred_dist, test_y):

    if isinstance(pred_dist, gpytorch.distributions.MultivariateNormal):
        return gpytorch.metrics.negative_log_predictive_density(pred_dist, test_y)
    else:
        log_probs = pred_dist.log_prob(test_y)
        assert(len(log_probs.shape) == 2)
        
        nr_samples = log_probs.shape[0]
        assert(nr_samples >= 100)

        # average over all monte carlo samples
        log_probs = log_probs.logsumexp(dim = 0) - math.log(nr_samples)

        # average over all data samples
        return - log_probs.mean()


def mean_standardized_log_loss(pred_dist, test_y):
    if isinstance(pred_dist, gpytorch.distributions.MultivariateNormal):
        return gpytorch.metrics.mean_standardized_log_loss(pred_dist, test_y)
    else:
        return torch.tensor(float("nan"))


def root_mean_squared_error(meanPredictions, test_y):
    assert(len(meanPredictions.shape) == 1)
    assert(meanPredictions.shape[0] == test_y.shape[0])

    res = torch.square(meanPredictions - test_y).mean()
    return res**0.5

def median_absolute_error(meanPredictions, test_y):
    diffs = torch.abs(meanPredictions - test_y)
    return torch.median(diffs)
