# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# %matplotlib inline

import math
import webdataset as wds
import scipy.ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import ray
from ray.data import Dataset
import braceexpand
import requests
import random
import json

ray.init()

# !gsutil ls gs://ocro-iaa/lin/ | shardsum

source = list(braceexpand.braceexpand("gs://ocro-iaa/lin/lin-{000000..000703}.tar"))
destination = "gs://ocro-tempout"

for s in source:
    # !gsutil cat $s | tar tvf - | sed 5q
    break

# !env | grep KEYS

dataset = ray.data.read_webdataset(source, parallelism=9999)


# +
def page_classifier(lin, do_plot=False, **kw):
    ys = []
    for text, bbox in lin:
        if bbox is None: continue
        bbox = map(int, bbox.split())
        x0, y0, x1, y1 = bbox
        ys.append(float(y1))
    if len(ys) < 10:
        return "mostly-empty"
    xs = np.linspace(0, 1, len(ys))
    ys = ndi.median_filter(ys, 5)
    ys = ndi.gaussian_filter(ys, 20.0)
    ys -= np.amin(ys)
    if np.amax(ys) < 1e-3:
        return "mostly-empty"
    ys /= np.amax(ys)
    ysm = np.maximum.accumulate(ys)
    delta = ys - ysm
    single = np.all(delta > -0.1)
    if single:
        result = "single"
    elif np.mean(ys[:len(ys)//4]) > 0.5:
        result = "backwards"
    else:
        result = "multi"
    color = dict(single="green", backwards="red", multi="blue")[result]
    if do_plot:
        plt.plot(xs, ys, color=color, **kw)
    return result

def balance_samples(sample):
    page = sample["lin.json"]
    kind = page_classifier(page)
    if kind == "single":
        if random.uniform(0, 1) < 0.1:
            return True
        else:
            return False
    elif kind == "multi":
        return True
    else:
        return False


# -

filtered_dataset = dataset.filter(balance_samples)

shuffled_dataset = filtered_dataset.random_shuffle()

shuffled_dataset.write_webdataset("gs://ocro-tempout/")
