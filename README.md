
# Robust Variational Gaussian Process Regression for Count Data with the Trimmed Marginal Likelihood

Implementation of the Trimmed Marginal Likelihood GP for regression of count data as proposed in [Robust Variational Gaussian Process Regression for Count Data with the Trimmed Marginal Likelihood](https://link.springer.com/article/10.1007/s11222-026-10895-9).


## Requirements

- Python >= 3.12
- PyTorch >= 2.0.1
- GPyTorch >= 1.13


## Preparation

1. Create experiment environment using e.g. conda as follows
```bash
conda create -n GPs python=3.12
conda activate GPs
pip3 install gpytorch matplotlib
```

2. Create nessecary folders for saving pre-processed data and results
```bash
mkdir all_results && mkdir all_results_hyper_params && mkdir all_summary_data && mkdir all_plots && mkdir openDatasets_prepared
```

3. (OPTIONAL - Only necessary for experiments with real data)
```bash
mkdir openDatasets_raw
```
Place the real data in *openDatasets_raw* and run *prepare_real_count_data.py*

4. Create synthetic data and/or add outliers to real data
```bash
python prepare_splits_add_outliers_count_data.py
```

## Run Experiments and Evaluate Results

-------------------------------------------
1. Run Experiments
-------------------------------------------

Run proposed method $\nu$-GP with Negative Binomial (NB) likelihoodon synthetic data with 5% outliers (without inducing points):
```bash
python runExperiments.py --likelihood=NB --method=trimmedLB --noise_type=lowest --true_outlier_ratio=0.05 
```

Same as above but using 200 inducing points:
```bash
python runExperiments.py --likelihood=NB --method=trimmedLB --noise_type=lowest --true_outlier_ratio=0.05 --reduced_rank=200 --learn_inducing_points=True 
```

Run ordinary count GP with Negative Binomial likelihood:
```bash
python runExperiments.py --likelihood=NB --method=variationalApprox --noise_type=lowest --true_outlier_ratio=0.05
``` 

Run OLRE (observation-level random effect) model with Poisson likelihood:
```bash
python runExperiments.py --likelihood=Poisson --method=OLRE --noise_type=lowest --true_outlier_ratio=0.05
``` 

Run $w$-GP model with NB likelihood:
```bash
python runExperiments.py --likelihood=NB --method=wGP --noise_type=lowest --true_outlier_ratio=0.05
``` 

Run GP with Poisson Likelihood trained by optimizing the $\gamma$-divergence ($\gamma$-div):
```bash
python runExperiments.py --likelihood=RobustPoisson --method=variationalApprox --noise_type=lowest --true_outlier_ratio=0.05
``` 

Run GP with NB Likelihood and post-hoc trimming (Post-Hoc) where $\nu$ is set to 0.2:
```bash
python runExperiments.py --likelihood=NB --method=variationalApproxPostHocTrimming --pre_specified_nu=0.2 --noise_type=lowest --true_outlier_ratio=0.05
``` 

For more details see argument description in *runExperiments.py*
All results for analysis are saved into folder "all_results/".

-------------------------------------------
3. Show summary of all results
-------------------------------------------

Determine outlier ratio (using Algorithm 2 and 3) and create summary data by running:
```bash
python create_summary_data.py --noise_type=lowest --true_outlier_ratio=0.05
``` 

Shows negative log-likelihood (nll) results:
```bash
python show_summary.py
``` 

Shows scaled continuous ranked probability scores (SCRPS):
```bash
python show_summary.py --scrps 
```

Shows outlier estimates:
```bash
python show_outlier_estimates.py
```


Plots the difference of a model's predictive cumulative distribution function (CDF) and the empirical CDF:
```bash
python create_plots.py
```

## Citation

If you are using part of the code in your work please cite the following paper:

Andrade, Daniel. "Robust Variational Gaussian Process Regression for Count Data with the Trimmed Marginal Likelihood." Statistics and Computing  36, 139 (2026): https://doi.org/10.1007/s11222-026-10895-9
