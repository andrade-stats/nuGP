import numpy as np
import scipy.stats
import torch

import commonSettings


def get_NB_log_probs(y_true, y_mean_preds, log_kappa):

    
    kappa = torch.exp(log_kappa) 
    
    latent_f = torch.log(y_mean_preds)

    logits = latent_f + torch.log(kappa)
    neg_binomial = torch.distributions.negative_binomial.NegativeBinomial(total_count = 1.0 / kappa, logits = logits)
    nll = - torch.mean(neg_binomial.log_prob(y_true))
    
    return nll
    

def get_NB_log_probs_scipy(y_true, y_mean_preds, kappa):
    
    latent_f = torch.log(y_mean_preds)

    logits = latent_f + torch.log(kappa)

    n = 1.0 / kappa
    p = n / (n + y_mean_preds)
    nb = scipy.stats.nbinom(n.detach().numpy(), p.detach().numpy())
    
    log_probs = np.log(nb.pmf(y_true.detach().numpy()))

    return - np.mean(log_probs)

    # print("probs = ", probs)
    # assert(False)
    # # neg_binomial = torch.distributions.negative_binomial.NegativeBinomial(total_count = 1.0 / kappa, logits = logits)
    # # nll = - torch.mean(neg_binomial.log_prob(y_true))
    # return nll


def get_NB_p_values(y_true, y_mean_preds, kappa):
    assert(y_true.shape == y_mean_preds.shape)
    
    n = 1.0 / kappa
    p = n / (n + y_mean_preds)
    nb = scipy.stats.nbinom(n, p)
    
    observations = y_true
    all_probs = np.stack((nb.cdf(observations), nb.sf(observations) + nb.pmf(observations)))
    p_values = np.min(all_probs, axis = 0)
    
    return p_values, nb.pmf(observations)
    



def get_best_kappa_nll(y_true_np, y_mean_preds_np, initial_kappa = 1.0):
    
    y_true = commonSettings.getTorchTensor(y_true_np)
    y_mean_preds = commonSettings.getTorchTensor(y_mean_preds_np)

    INIITAL_LOG_KAPPA = np.log(initial_kappa)
    TRAINING_ITR = 200
    LEARNING_RATE = 0.1

    log_kappa = torch.nn.parameter.Parameter(data = torch.tensor([INIITAL_LOG_KAPPA]))

    optimizer = torch.optim.Adam([log_kappa], lr=LEARNING_RATE) # need to include likelihood explicitly
    
    for i in range(TRAINING_ITR):
        optimizer.zero_grad()
        nll = get_NB_log_probs(y_true, y_mean_preds, log_kappa) 
        nll.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"iteration {i}: kappa = {torch.exp(log_kappa)}, nll = {nll}")

   
    kappa = torch.exp(log_kappa).item() 
    best_nll = nll.item() 

    return kappa, best_nll

