
import numpy
import torch

# Data from J.H. Friedman, Multivariate adaptive regression splines, Ann. Stat. 19 (1) (1991) 1–67.
# f exactly as in "Robust Regression with twinned Gaussian Processes", page 7
# but y = NegativeBinomial(logits = f, kappa)
def getFriedmanCountData(n, kappa, nrRepetitions):
    nrVariables = 10

    RANDOM_GENERATOR_SEED = 9899832
    numpy.random.seed(RANDOM_GENERATOR_SEED)
    
    allX = []
    allY = []
    
    for repetitionId in range(nrRepetitions):
        
        X = numpy.random.rand(n, nrVariables)
        latent_f = 1.0 * numpy.sin(numpy.pi * X[:,0] * X[:,1]) + 2.0 * numpy.square(X[:,2] - 0.5) + 1.0 * X[:,3] + 0.5 * X[:,4]
        
        latent_f = torch.from_numpy(latent_f)
        print("latent_f = ", latent_f.shape)

        kappa = torch.tensor(kappa)
        logits = latent_f + torch.log(kappa)
        NB = torch.distributions.negative_binomial.NegativeBinomial(total_count = 1.0 / kappa, logits = logits)

        y = NB.sample([1]).squeeze()
        y = y.numpy()

        # print("X = ", X.shape)
        # print("y = ", y.shape)
        # print("y = ", y[0:20])
        
        X = X.astype(numpy.float32)
        y = y.astype(numpy.float32)

        print("synthetic data:")
        print("variance(y) = ", numpy.var(y))
        print("mean(y) = ", numpy.mean(y))

        allX.append(X)
        allY.append(y)
    
    return allX, allY