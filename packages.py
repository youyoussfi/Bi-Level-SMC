# packages Particles
import particles
from particles import distributions as dists
from particles import resampling as rs
from particles import smc_samplers as ssps
from particles import binary_smc as bin

import os, re, operator, calendar, functools, math
import random, ast, pickle, tqdm, warnings
import pandas as pd
import numpy as np

import time
from collections import Counter
from functools import reduce
from dateutil import rrule
import itertools
from itertools import compress, product 
import time as t
import datetime as dt
from datetime import datetime

import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

import scipy
from scipy import optimize, stats
from scipy.stats import norm, chi2_contingency
from scipy.spatial.distance import cosine
from statsmodels.discrete.discrete_model import Probit
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel