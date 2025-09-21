#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
from math import log
from pathlib import Path
from time import time
from typing import Any
from types import SimpleNamespace
import uuid


import numpy as np
from multiprocessing import Pool
from scipy.special import logsumexp, gammaln

# from tqdm import tqdm
from tqdm.notebook import tqdm


# ## Random samples

# In[2]:


info = {
    "MNIST": {
        "VAE": "where is save results of model (files *.npz)",
        "IWAE(5)": "where is save results of model (files *.npz)",
        "IWAE(10)": "where is save results of model (files *.npz)",
    },
    "SVHN": {
        "VAE": "where is save results of model (files *.npz)",
        "IWAE(5)": "where is save results of model (files *.npz)",
        "IWAE(10)": "where is save results of model (files *.npz)",
    },
    "CelebA": {
        "VAE": "where is save results of model (files *.npz)",
        "IWAE(5)": "where is save results of model (files *.npz)",
        "IWAE(10)": "where is save results of model (files *.npz)",
    },
}


# In[3]:


args = SimpleNamespace(
    dataname="MNIST",
    modelname="VAE",
    num=7,
    n_jobs=10,
    optimal_sigma=0.3,
    output="results.npy",
    num_images=10000,
    max_features=None,
    root_save="root_save"
)

print(args)


# In[4]:


# ===================================================
root_data = Path(info[args.dataname][args.modelname])
files = list(root_data.glob("**/*.npz"))
# ===================================================

memory_str = ("B", "KB", "MB", "GB")
arr = None

for file in (bar := tqdm(files, total=len(files))):
    with np.load(file) as data:
        # Ładowanie danych raz i buforowanie
        values_I = data["values2bound_I"][:args.num_images]
        values_II = data["values2bound_II"][:args.num_images]
        
        current_data = np.concatenate([values_I, values_II], axis=1)
        
        if arr is None:
            arr = current_data
        else:
            arr = np.concatenate([current_data, arr], axis=1)
    
    memory_bytes = arr.nbytes
    for i, unit in enumerate(memory_str):
        if memory_bytes < 1024.0 * (1000 if i > 0 else 1) or i == 3:
            memory_size = memory_bytes / (1024.0 ** i) if i > 0 else memory_bytes
            break
        memory_bytes /= 1024.0
    
    bar.set_description(f"Raw data: {memory_size:.4f} {unit} | shape: {arr.shape}")
    
    if args.max_features is not None and arr.shape[-1] > args.max_features:
        break


# In[5]:


def bootstrapping(data: np.ndarray, n_feature: int, identifier: int) -> Any:
    np.random.seed(seed=identifier)
    idxes = np.random.choice(data.shape[1], size=n_feature, replace=True)
    return logsumexp(data[:, idxes], axis=1, keepdims=True) - log(float(n_feature))


def uuid_filename():
    return str(uuid.uuid4())[:12]


# In[9]:


args.root_save = Path(args.root_save)
args.root_save.mkdir(parents=True, exist_ok=True)
print(arr)

indexes = np.arange(arr.shape[0])
results = {"n": [], "time": [], **{i: [] for i in indexes}}
arr_res = np.full((args.num_images, args.num), np.nan)

features = (
    *range(25, 101, 25),
    *range(150, 251, 50),
    *range(300, 501, 100),
    *range(1000, 5001, 1000),
    *range(10000, 50001, 10000),
    *range(100000, 500001, 100000),
    750000, 1000000, 2000000, 5000000
)
for N_FEATURE in (bar := tqdm(features, desc="Searching for the number of features")):
    start_time = time()
    with Pool(processes=args.n_jobs) as pool:
        log_x = pool.starmap(
            bootstrapping, [(arr, N_FEATURE, i) for i in range(args.num)]
        )
    log_x = np.concatenate(log_x, axis=1)
    elapsed_time = time() - start_time

    results["n"].append(N_FEATURE)
    results["time"].append(elapsed_time)

    sigma = np.std(log_x, axis=1)
    for i, j in enumerate(indexes):
        results[j].append(sigma[i])

    idx = sigma > args.optimal_sigma
    arr_res[indexes[~idx]] = log_x[~idx]

    if idx.any():
        arr = arr[idx]
        indexes = indexes[idx]

    # np.save(args.output, results, allow_pickle=True)  # uncomment if you want to save
    
    if np.max(sigma) < args.optimal_sigma:
        break
    bar.set_description(
        f"Features: {N_FEATURE}|Remaining: {indexes.size}|Time: {elapsed_time:.2f}s"
    )
    
if np.isnan(arr_res).any():
    arr_res = arr_res[~np.any(np.isnan(arr_res), axis=1)]
    
# np.save(args.root_save / f"{uuid_filename()}.npy", arr_res)  # uncomment if you want to save


# ## Calcluate bounds

# In[7]:


k_data = [2, 3]
ncols = 2 * len(k_data)

log_x = arr_res
outputs = np.zeros(ncols, dtype=log_x.dtype)

X = log_x[:, :1]
for ii, k in enumerate(k_data):
    Y = log_x[:, 1 : 2 * k]
    
    j = np.arange(1, 2 * k)
    binomial_coefficients = gammaln(2 * k) - gammaln(j + 1) - gammaln(2 * k - j)
    results = (
        binomial_coefficients.reshape(1, -1)
        + np.cumsum(Y - X, axis=1)
        - np.log(j)[None, :]
    )
    x = logsumexp(results[:, ::2], axis=1)
    if k == 1:
        outs = np.exp(x)
    else:
        v = logsumexp(results[:, 1::2], axis=1)
        outs = np.exp(x) - np.exp(v)
        
    assert np.all(np.isfinite(outs))
        
    if k == 1:
        outputs[2 * ii] = log(np.mean(outs - np.sum(1 / j), axis=0) + 1)
        continue
    
    log_mu = np.mean(Y)
    tmp = log_mu - X
    # lower bound, not additive
    results = (
        binomial_coefficients.reshape(1, -1) - j[None, :] * tmp - np.log(j)[None, :]
    )
    x = logsumexp(results[:, ::2], axis=1)
    if k == 1:
        outs = -np.exp(x)
    else:
        v = logsumexp(results[:, 1::2], axis=1)
        outs = np.exp(v) - np.exp(x)
    assert np.all(np.isfinite(outs))
    
    selected_temp = outs + np.sum(1 / j)
    outputs[2 * ii] = np.mean(selected_temp, axis=0)    # lower non additive
    
    # upper bound, not additive
    results = (
        binomial_coefficients.reshape(1, -1) + j[None, :] * tmp - np.log(j)[None, :]
    )
    x = logsumexp(results[:, ::2], axis=1)
    if k == 1:
        outs = np.exp(x)
    else:
        v = logsumexp(results[:, 1::2], axis=1)
        outs = np.exp(x) - np.exp(v)
    assert np.all(np.isfinite(outs))
    outputs[2 * ii + 1] = np.mean(outs - np.sum(1 / j), axis=0)  # upper non additive


# In[8]:


print(f"k2 lower non additive: {outputs[0]:.5g}")
print(f"k2 upper non additive: {outputs[1]:.5g}")
print(f"k3 lower non additive: {outputs[2]:.5g}")
print(f"k3 upper non additive: {outputs[3]:.5g}")


# In[ ]:





# In[ ]:





# In[ ]:




