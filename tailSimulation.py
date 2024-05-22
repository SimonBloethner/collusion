import numpy as np
from datetime import datetime
from statsmodels.api import WLS
import os


def bp_mean(alpha, l, h):
    if alpha == 1:
        return ((h * l) / (h - l)) * np.log(h / l)
    else:
        mn = (l ** alpha) / (1 - (l / h) ** alpha) * (alpha / (alpha - 1)) * ((1 / l ** (alpha - 1)) - (1 / (h ** (alpha - 1))))
        return mn


def bounded_pareto(size, alpha, lowers, uppers):
    u = np.random.uniform(0, 1, size=size)
    numerator = u * (uppers ** alpha) - u * (lowers ** alpha) - (uppers ** alpha)
    denominator = (uppers ** alpha) * (lowers ** alpha)
    base = -(numerator / denominator)
    bp = np.sign(base) * np.abs(base) ** (- 1 / alpha)
    return bp


start = datetime.now()

np.random.seed(13)

num_trials = 1000
num_inputs = 2
num_obs = 2 * 10 ** 5

strg_ests = np.zeros((num_trials, num_inputs))
strg_sampling_ests = np.zeros((num_trials, num_inputs))
strg_sds = np.zeros((num_trials, num_inputs))
strg_sampling_sds = np.zeros((num_trials, num_inputs))
strg_betas = np.zeros((num_trials, num_inputs))
strg_means = np.zeros((num_trials, 1))
strg_emp_means = np.zeros((num_trials, 1))
strg_alphas = np.zeros((num_trials, num_inputs))
strg_lowers = np.zeros((num_trials, num_inputs))
strg_uppers = np.zeros((num_trials, num_inputs))
iterations = np.zeros((num_trials, 1))
req_samples = np.zeros((num_trials, num_inputs))

for trial in range(num_trials):
    alphas = np.random.uniform(0.9, 2.5, num_inputs)
    # betas = np.random.uniform(0, 50, num_inputs)
    lowers = np.random.uniform(10**-6, 10**-3, num_inputs)
    uppers = np.random.uniform(10**0, 10**2, num_inputs)

    betas = np.zeros((num_obs, num_inputs))

    for var in range(num_inputs):
        betas[:, var] = bounded_pareto(num_obs, alphas[var], lowers[var], uppers[var])
        strg_betas[trial, var] = bp_mean(alphas[var], lowers[var], uppers[var])

    strg_alphas[trial, :] = alphas
    strg_lowers[trial, :] = lowers
    strg_uppers[trial, :] = uppers

    strg = np.random.poisson(10 ** 5, size=num_obs * num_inputs).reshape(num_obs, num_inputs)
    means = np.zeros((num_inputs, 1))
    out = np.zeros(num_obs)

    for var in range(num_inputs):
        out += strg[:, var] * betas[:, var]

    est_data = np.c_[out, strg]

    strg_emp_means[trial, :] = np.mean(est_data[:, 0])
    theo_mean = strg_betas[trial, :] @ strg.mean(axis=0)

    strg_means[trial] = theo_mean

    wls_model = WLS(endog=est_data[:, 0], exog=est_data[:, 1:], weights=1).fit()
    res = wls_model.params
    sds = wls_model.bse

    strg_ests[trial, :] = res
    strg_sds[trial, :] = sds

    dep = est_data[:, 0]

    body_mean = np.mean(dep[dep < theo_mean]) * (dep[dep < theo_mean].shape[0] / dep.shape[0])
    tail_mean = np.mean(dep[dep > theo_mean]) * ((dep[dep > theo_mean]).shape[0] / dep.shape[0])

    weight = np.ones(est_data.shape[0])
    weight[est_data[:, 0] > theo_mean] = (theo_mean - body_mean) / tail_mean

    wls_model = WLS(endog=est_data[:, 0], exog=est_data[:, 1:], weights=weight).fit()
    res = wls_model.params
    sds = wls_model.bse

    strg_sampling_ests[trial, :] = res
    strg_sampling_sds[trial, :] = sds


mean_diff = strg_means - strg_emp_means
ests_diff = strg_ests[:, 1:] - strg_betas
samp_ests_diff = strg_sampling_ests[:, 1:] - strg_betas
ranges = strg_uppers.max(axis=1) - strg_lowers.min(axis=1)
alphas = np.min(strg_alphas, axis=1)

dfs = ['mean_diff', 'ests_diff', 'samp_ests_diff', 'ranges', 'strg_alphas', 'strg_sds', 'strg_sampling_sds']

if str(os.path).find('Simon') > 0:
    location = '/Users/Simon/Documents/Projects/EWF VI/Research/PhD/Tails/data'
else:
    location = 'tradeTails/outData'


for df in dfs:
    np.savetxt('{}/{}.csv'.format(location, df), eval(df), delimiter=',')

end = datetime.now()
print('Start time was: {}'.format(start))
print('End time was: {}'.format(end))
print('Execution time was {}'.format(datetime.now() - start))

