import math
import torch
import gpytorch
import numpy as np

import metrics_GP
import scipy.stats
from scipy.cluster.vq import kmeans2

import commonSettings
from commonSettings import EstimationType

import time

# important hyper-parameters

L_MIN_STANDARD_NR_TRAINING_ITERATIONS = 1000
L_MAX_STANDARD_NR_TRAINING_ITERATIONS = 50000

L_LEARNING_RATE = 0.01

ZERO_MEAN_FUNCTION = gpytorch.means.ZeroMean()

CONST_MEAN_FUNCTION = gpytorch.means.ConstantMean() # use this for count data, since count data (y) is not normalized

LOWER_BOUND_ON_SIGMA = math.sqrt(1.000E-04)   # set by gpytorch
INITIAL_SIGMA_SQUARE = 10.0



def getCovFunc(covFunc_name, d):
    if covFunc_name == "SE":
        # squared exponential (SE) kernel  
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = d))
    elif covFunc_name == "Matern":
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 2.5, ard_num_dims = d))
    else:
        assert(False)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,  meanFunc, covFunc):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = meanFunc
        self.covar_module = covFunc
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, meanFunc, covFunc, reducedRank, learn_inducing_points):

        if reducedRank is None:
            # use all training data points
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(train_x.shape[0])
            
            # sets inducing_points to train_x
            variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
                model = self, inducing_points = train_x, variational_distribution = variational_distribution, learn_inducing_locations=False
            )
        else:
            assert(learn_inducing_points is not None)
            assert(reducedRank >= 10 and reducedRank <= 5000 and reducedRank < train_x.shape[0])
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(reducedRank)

            # as in https://docs.gpytorch.ai/en/stable/examples/05_Deep_Gaussian_Processes/Deep_Sigma_Point_Processes.html
            inducing_points = train_x[torch.randperm(train_x.shape[0])[0:reducedRank], :]
            inducing_points = inducing_points.clone().data.cpu().numpy()
            inducing_points = commonSettings.getTorchTensor(kmeans2(train_x.data.cpu().numpy(), inducing_points, minit='matrix')[0])
            assert(inducing_points.shape[0] == reducedRank and inducing_points.shape[1] == train_x.shape[1])

            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_points
            )


        super(VariationalGPModel, self).__init__(variational_strategy)

        self.mean_module = meanFunc
        self.covar_module = covFunc
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred



class BasicGP:
    def __init__(self, model, likelihood):
        self.model = model
        self.likelihood = likelihood
        
    def getMeanPrediction(self, X):

        if len(X.shape) == 1:
            X = X.view(1, -1)
        
        predictive_distribution_at_X = getPredictions(self.model, self.likelihood, X)
        return getMeanPredictions(predictive_distribution_at_X)
    

    def aggregateSamples(eval_result):
        if len(eval_result.shape) == 2:
            print("RUN MEAN")
            return torch.mean(eval_result, dim = 0)
        else:
            return eval_result

    # returns log p(y_i | x_i) for each i independently (instead of using p(y_1, ..., y_n | X))
    def get_all_log_probs_ind(self, X, true_y):
        pred_dist = getPredictions(self.model, self.likelihood, X)

        if isinstance(pred_dist, gpytorch.distributions.MultivariateNormal):
            all_means = pred_dist.loc
            all_scales = torch.diag(pred_dist.scale_tril)
            pred_dist_as_normal = torch.distributions.normal.Normal(loc = all_means, scale = all_scales)
            log_probs = pred_dist_as_normal.log_prob(true_y)
        else:
            log_probs = pred_dist.log_prob(true_y)
            assert(len(log_probs.shape) == 2)
            
            nr_mc_samples = log_probs.shape[0]
            assert(nr_mc_samples > 100)

            # average over all monte carlo samples
            log_probs = log_probs.logsumexp(dim = 0) - np.log(nr_mc_samples)

        return log_probs.detach().cpu().numpy()
            

    def get_one_sided_p_value_NB(self, X, true_y):
        pred_dist = getPredictions(self.model, self.likelihood, X)

        MC_SAMPLES = 200
        assert(pred_dist.param_shape[0] == MC_SAMPLES)
        assert(isinstance(pred_dist, torch.distributions.negative_binomial.NegativeBinomial))

        n = pred_dist.total_count
        p = n / (n + pred_dist.mean)
        
        nb = scipy.stats.nbinom(n.cpu().numpy(), p.cpu().numpy())
        
        observed_y = true_y.cpu().numpy()
        all_right_sided_p_values_all_samples = nb.sf(observed_y) + nb.pmf(observed_y)
        all_left_sided_p_values_all_samples = nb.cdf(observed_y)

        all_left_sided_p = np.mean(all_left_sided_p_values_all_samples, axis = 0)
        all_right_sided_p = np.mean(all_right_sided_p_values_all_samples, axis = 0)
        
        return all_left_sided_p, all_right_sided_p
    
    
    def getAbsResiduals_and_MeanPredictions(self, X, true_y):
        predictive_distribution_at_X = getPredictions(self.model, self.likelihood, X)
        meanPredictions = getMeanPredictions(predictive_distribution_at_X)
        residuals = torch.abs(meanPredictions - true_y)
        return residuals.detach().cpu().numpy(), meanPredictions.detach().cpu().numpy()
    
    
    def evaluatePredictions(self, X, true_y):
        
        if len(X.shape) == 1:
            X = X.view(1, -1)
        
        predictive_distribution_at_X = getPredictions(self.model, self.likelihood, X)
        
        # print("predictive_distribution_at_X = ", predictive_distribution_at_X)
        # assert(False)

        print("calculate evaluation measures:")
        nll = metrics_GP.negative_log_predictive_density(predictive_distribution_at_X, true_y) # average negative log likelihood
        
        log_probs = self.get_all_log_probs_ind(X, true_y)
        assert(log_probs.shape[0] == true_y.shape[0] and len(log_probs.shape) == 1)
        nll_ind_mean = np.mean(- log_probs)
        nll_ind_median = np.median(- log_probs)

        msll = metrics_GP.mean_standardized_log_loss(predictive_distribution_at_X, true_y) # as in "Gaussian Processes for Machine Learning", page 23 (41 pdf)

        meanPredictions = getMeanPredictions(predictive_distribution_at_X)
        rmse = metrics_GP.root_mean_squared_error(meanPredictions, true_y)
        median_absolute_error = metrics_GP.median_absolute_error(meanPredictions, true_y)   

        if type(predictive_distribution_at_X) is torch.distributions.negative_binomial.NegativeBinomial:
            # need to be careful about the parameterization in scipy (p = 1.0 - predictive_distribution_at_X.probs)
            predictive_distribution_at_X_scipy = scipy.stats.nbinom(n = predictive_distribution_at_X.total_count.cpu().detach().numpy(), p = 1.0 - predictive_distribution_at_X.probs.cpu().detach().numpy())
        elif type(predictive_distribution_at_X) is torch.distributions.poisson.Poisson:
            predictive_distribution_at_X_scipy = scipy.stats.poisson(mu = predictive_distribution_at_X.rate.cpu().detach().numpy())
        else:
            assert(False)
        
        # checked
        # log_probs_pytorch = predictive_distribution_at_X.log_prob(true_y)
        # log_probs_scipy = predictive_distribution_at_X_scipy.logpmf(true_y.cpu().detach().numpy())
        # print("nb = ", nb)
        # print("log_probs_pytorch = ", log_probs_pytorch)
        # print("log_probs_scipy = ", log_probs_scipy)
        # assert(False)
        
        return nll, msll, rmse, median_absolute_error, nll_ind_mean, nll_ind_median, predictive_distribution_at_X_scipy


    def getMLL(self, X_new = None, y_new = None):
        assert(isinstance(self.likelihood, gpytorch.likelihoods._GaussianLikelihoodBase))

        if X_new is not None:
            self.model.set_train_data(inputs=X_new, targets=y_new, strict = False)

        self.model.train()
        self.likelihood.train()

        assert(self.model.training and self.likelihood.training)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        mll = mll.to(device=commonSettings.DEVICE)

        X = self.model.train_inputs[0]
        y = self.model.train_targets
        return mll(self.model(X), y).item()





# checked
def getPredictions(model, likelihood, X_new):

    model.eval()
    likelihood.eval()
    
    # number of samples used for estimating the integral of the liklihood =  int_f p(y | f) p(f) df,
    # where p(f) is a multivariate gaussian, and p(y | f) is the likelihood (e.g. student t)
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(200):
        variational_approximation = model(X_new) # here model(X_new) returns the Normal approximation of p(f_new | x_new, y_train, X_train), for details see e.g. forward method of UnwhitenedVariationalStrategy
        predictive_distribution = likelihood(variational_approximation)   
        
    return predictive_distribution


# get E[y | x]
def getMeanPredictions(predictive_distribution):
    if isinstance(predictive_distribution, torch.distributions.negative_binomial.NegativeBinomial) or isinstance(predictive_distribution, torch.distributions.poisson.Poisson):
        meanPredictions = predictive_distribution.mean
        assert(len(meanPredictions.shape) == 2)
        meanPredictions = torch.mean(meanPredictions, axis = 0)
        return meanPredictions
    else:
        meanPredictions = predictive_distribution.loc.detach()
        if len(meanPredictions.shape) == 2:
            meanPredictions = torch.mean(meanPredictions, axis = 0)
        return meanPredictions



# checked
def showProgressGP(i, loss, model, likelihood, startTime):
    # numpy.set_printoptions(precision=2)
    lengthscaleOutput = model.covar_module.base_kernel.lengthscale.cpu().detach().numpy()[0,:]

    # note that likelihood.noise.item() correponds to sigma^2

    if (i <= 100) or (i % 100 == 0):
        runtime = time.time() - startTime
        if hasattr(likelihood, 'deg_free'):
            # student-t likelihood
            print(f"Iter {i} - Loss: {loss.item():.3f} outputscale: {model.covar_module.outputscale.item():.3f}    lengthscale: {lengthscaleOutput}   noise variance: {likelihood.noise.item():.3f}   nu: {likelihood.deg_free.item():.3f}   (runtime = {(runtime / 60.0):.3f})")
        elif hasattr(likelihood, 'kappa'):
            # Negative Binomial (NB) likelihood
            print(f"Iter {i} - Loss: {loss.item():.3f} outputscale: {model.covar_module.outputscale.item():.3f}    lengthscale: {lengthscaleOutput}   kappa: {likelihood.kappa.item():.3f}   (runtime = {(runtime / 60.0):.3f})")
        elif hasattr(likelihood, 'noise'):
            # Normal likelihood
            print(f"Iter {i} - Loss: {loss.item():.3f} outputscale: {model.covar_module.outputscale.item():.3f}    lengthscale: {lengthscaleOutput}   noise variance: {likelihood.noise.item():.3f}  (runtime = {(runtime / 60.0):.3f})")
        else:
            # Poisson likelihood
            print(f"Iter {i} - Loss: {loss.item():.3f} outputscale: {model.covar_module.outputscale.item():.3f}    lengthscale: {lengthscaleOutput}   (runtime = {(runtime / 60.0):.3f})")
        
    return

# checked
def standardTraining(args, model, likelihood, mll, X, y):

    # Use the adam optimizer
    if type(model) is ExactGPModel:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # Includes GaussianLikelihood parameters
    else:
        assert(type(model) is VariationalGPModel)
        optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": likelihood.parameters()}], lr=args.learning_rate) # need to include likelihood explicitly
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    previous_loss = float("inf")
    
    all_losses = np.zeros(args.max_training_itr) * np.nan
    
    startTime = time.time()

    for i in range(args.max_training_itr):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Calc loss and backprop gradients
        loss = -mll(model(X), y) # note: mll returns marginal log-likelihood divided by the number of samples
       
        loss.backward()
        optimizer.step()

        all_losses[i] = loss.detach().cpu().numpy()
        showProgressGP(i, loss, model, likelihood, startTime)
        
        if loss >= previous_loss and i >= args.min_training_itr:
            break
        else:
            previous_loss = loss
    
    
    average_mll_value = mll(model(X), y).item()
    marginalLikelihood_value = average_mll_value  * y.shape[0]
    return model, likelihood, marginalLikelihood_value, average_mll_value, all_losses




# ****************************************************************************
# ****************************************************************************
# ****************************************************************************



def classifyBasedOnScores(outlierScores, maxNrOutlierSamples):
    _, outlierIds = torch.sort(- outlierScores)[0:maxNrOutlierSamples]
    outliers_zeroOne = torch.zeros_like(outlierScores)
    outliers_zeroOne[outlierIds] = 1
    return outliers_zeroOne




# checked
# corresponds to Standard-GP in paper
def trainVanillaGP(X, y, sigmaEstimateTypes):

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X, y, likelihood)
    likelihood.noise = INITIAL_SIGMA_SQUARE

    model = commonSettings.setDataType(model)
    likelihood = commonSettings.setDataType(likelihood)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model, likelihood, _, average_mll_value = standardTraining(model, likelihood, mll, X, y)
    learnedGP = BasicGP(model, likelihood)

    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    standardizedResiduals = {}

    for sigmaEstimateType in sigmaEstimateTypes:
        
        # note that likelihood.noise.item() correponds to sigma^2
        estimatedNoiseVariance = likelihood.noise.item()

        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        meanPredictions = learnedGP.getMeanPrediction(X)
        centeredAbsValues = torch.abs(meanPredictions - y.detach())

        sigmaEstimate = math.sqrt(estimatedNoiseVariance)
        allPValues_logScale[sigmaEstimateType] = getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate)
        allEstimatedNoiseVariances[sigmaEstimateType] = estimatedNoiseVariance
        standardizedResiduals[sigmaEstimateType] = (centeredAbsValues / sigmaEstimate).cpu().numpy()

    return allPValues_logScale, allEstimatedNoiseVariances, learnedGP, average_mll_value, standardizedResiduals


# simple baseline that filters out large squared y-values
def LMD(X, y, maxNrOutlierSamples):
    nr_inliers = X.shape[0] - maxNrOutlierSamples
    allInlierSamplesIds = torch.argsort(torch.square(y))[0:nr_inliers]
    allOutlierSampleIds = torch.argsort(torch.square(y))[nr_inliers:y.shape[0]]
    assert(allInlierSamplesIds.shape[0] + allOutlierSampleIds.shape[0] == y.shape[0])
    _, _, gpModel, _, _= trainVanillaGP(X[allInlierSamplesIds, :], y[allInlierSamplesIds], sigmaEstimateTypes = commonSettings.getSigmaEstimationTypes())


    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    standardizedResiduals = {}

    for sigmaEstimateType in commonSettings.getSigmaEstimationTypes():
        
        # note that likelihood.noise.item() correponds to sigma^2
        estimatedNoiseVariance = gpModel.likelihood.noise.item()

        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        meanPredictions = gpModel.getMeanPrediction(X)
        centeredAbsValues = torch.abs(meanPredictions - y.detach())

        sigmaEstimate = math.sqrt(estimatedNoiseVariance)
        allPValues_logScale[sigmaEstimateType] = getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate)
        allEstimatedNoiseVariances[sigmaEstimateType] = estimatedNoiseVariance
        standardizedResiduals[sigmaEstimateType] = (centeredAbsValues / sigmaEstimate).cpu().numpy()

    return allInlierSamplesIds, allOutlierSampleIds, allPValues_logScale, allEstimatedNoiseVariances, gpModel, standardizedResiduals




# checked
def trainStudentTGP(X, y, sigmaEstimateTypes):

    # initialize likelihood and model
    model = VariationalGPModel(X)
    likelihood = gpytorch.likelihoods.StudentTLikelihood()
    likelihood.noise = INITIAL_SIGMA_SQUARE

    model = commonSettings.setDataType(model)
    likelihood = commonSettings.setDataType(likelihood)

    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y.shape[0])

    model, likelihood, _, average_mll_value = standardTraining(model, likelihood, mll, X, y)
    learnedGP = BasicGP(model, likelihood)

    allPValues_logScale = {}
    allEstimatedNoiseVariances = {}
    standardizedResiduals = {}

    for sigmaEstimateType in sigmaEstimateTypes:

        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        allPValues_logScale[sigmaEstimateType], allEstimatedNoiseVariances[sigmaEstimateType], standardizedResiduals[sigmaEstimateType] = getRobustEstimates(learnedGP, X, y)

    return allPValues_logScale, allEstimatedNoiseVariances, learnedGP, average_mll_value, standardizedResiduals


def getRobustEstimates(learnedGP, X, y):
    meanPredictions = learnedGP.getMeanPrediction(X)
    centeredAbsValues = torch.abs(meanPredictions - y.detach())
    
    correctionFactor = math.sqrt(1.0 / scipy.stats.chi2.ppf(0.5, df = 1.0))
    correctedSigmaEstimate = correctionFactor * torch.median(centeredAbsValues)
    correctedSigmaEstimate = correctedSigmaEstimate.item()
    logPValues = getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate = correctedSigmaEstimate)
    noiseVariance = correctedSigmaEstimate ** 2
    standardizedResiduals = centeredAbsValues / correctedSigmaEstimate

    return logPValues, noiseVariance, standardizedResiduals.cpu().numpy()


def getAllRobustEstimates(learnedGP, X, y):
    standardizedResiduals = {}

    for sigmaEstimateType in commonSettings.getSigmaEstimationTypes():
        assert(sigmaEstimateType == EstimationType.LIKELIHOOD_NOISE)
        _, _, standardizedResiduals[sigmaEstimateType] = getRobustEstimates(learnedGP, X, y)

    return standardizedResiduals


# checked
def getPValues_fromStudentT_model(predictions_at_trainingDataPoints, observed_y):
    centeredAbsValues = numpy.abs((predictions_at_trainingDataPoints.loc - observed_y).numpy())
    studentT = scipy.stats.t(loc = numpy.zeros_like(centeredAbsValues), df = predictions_at_trainingDataPoints.df.numpy(), scale = predictions_at_trainingDataPoints.scale.numpy())
    
    pValuesEachSample = 2.0 * studentT.cdf(- centeredAbsValues)
    pValues = numpy.mean(pValuesEachSample, axis = 0)

    return pValues



# checked
def getPValues_fromGP_model(predictions_at_trainingDataPoints, observed_y):

    allScales = (torch.sqrt(torch.diag(predictions_at_trainingDataPoints.covariance_matrix))).detach().numpy()

    centeredAbsValues = numpy.abs((predictions_at_trainingDataPoints.loc - observed_y).detach().numpy())
    normal = scipy.stats.norm(loc = numpy.zeros_like(centeredAbsValues), scale = allScales)
    
    pValues = 2.0 * normal.cdf(- centeredAbsValues)

    return pValues




def getLogPValues_fromSigmaEstimate_and_absValues(centeredAbsValues, sigmaEstimate):
    normal = scipy.stats.norm(loc = numpy.zeros_like(centeredAbsValues.cpu().numpy()), scale = sigmaEstimate)
    log_pValues = math.log(2.0) + normal.logcdf(- centeredAbsValues.cpu().numpy())
    log_pValues = torch.from_numpy(log_pValues)
    log_pValues = log_pValues.float()
    log_pValues = log_pValues.to(device=commonSettings.DEVICE)
    return log_pValues

# estimates the scale (sigma) of the random noise, which is assumed to be gaussian
def getCorrectedSigmaEstimate(likelihood, predictions_at_trainingDataPoints, observed_y, maxNrOutlierSamples = None):

    observed_y = observed_y.detach()

    # get E[y | x]
    meanPredictions = getMeanPredictions(predictions_at_trainingDataPoints)
    

    if type(likelihood) is gpytorch.likelihoods.GaussianLikelihood:
        if maxNrOutlierSamples is None:
            # no correction needed
            estimatedNoiseVariance = likelihood.noise.item()
            correctedSigmaEstimate = numpy.sqrt(estimatedNoiseVariance)
            
            estimatedVarFromPredictions = numpy.mean(numpy.square(meanPredictions - observed_y))
            print("estimatedVarFromPredictions = ", estimatedVarFromPredictions)
            print("estimatedNoiseVariance = ", estimatedNoiseVariance)
        else:
            # use asymptotic correction
            centeredAbsValues = numpy.abs(meanPredictions - observed_y)
            n = observed_y.shape[0]
            inlierAbsDiff = numpy.sort(centeredAbsValues)[0:(n - maxNrOutlierSamples)]
            correctedSigmaEstimate = getAsymptoticCorrectedSigma(inlierAbsDiff, n) 
    else:
        assert(type(likelihood) is gpytorch.likelihoods.StudentTLikelihood)
        # use MAD (median absolute deviation) with asymptotic correction
        centeredAbsValues = numpy.abs(meanPredictions - observed_y)
        correctionFactor = numpy.sqrt(1.0 / scipy.stats.chi2.ppf(0.5, df = 1.0))
        correctedSigmaEstimate = correctionFactor * numpy.median(centeredAbsValues)
        
    return correctedSigmaEstimate


# checked (from sigmaCorrectionMethods.py)
def getAsymptoticCorrectedSigma(inlierAbsDiff, n):

    m = inlierAbsDiff.shape[0]
    inlierRatio = m / n

    if m == n:
        return torch.sqrt(torch.mean(torch.square(inlierAbsDiff))).item()
    else:
        correctionFactor = 1.0 / scipy.stats.chi2.ppf(inlierRatio, df = 1.0)

        empiricalQuantile = torch.max(inlierAbsDiff)
        
        correctedSigma = empiricalQuantile * torch.sqrt(correctionFactor)
        return correctedSigma.item()