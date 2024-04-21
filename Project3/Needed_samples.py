import pandas as pd
import numpy as np
import scipy.stats
import os

os.chdir("C:/Users/cheli/Downloads")

data = pd.read_csv("data_1200.csv")
del data[data.columns[0]]

# Examinating how many samples are nessesary 
# Function for calculating 95% confidence interval
def confidence_interval(data, confidence=0.95):
    data = np.array(data) # make sure data is numpy array
    n = len(data) # sample size
    m, se = np.mean(data), scipy.stats.sem(data) # sample mean and standard error 
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1) # margin of error
    return m-h, m+h # return lower and upper confidence 

# Function for determine whether the data is different
def difference_num(num):
    # Start by assuming the is no significant difference
    different = False
    # Try 100 different sub-samplings
    for n in range(100):
        # sample 10 random data points from our "bank"
        sampled_data = data.sample(num)
        # Loop through the variables in the data
        for variable in data:
            # Determine lower and upper confidence bound for original and sub-samled data
            lower1, upper1 = confidence_interval(data[variable])
            lower2, upper2 = confidence_interval(sampled_data[variable])
            # If the confidence bound do not overlap, the data is significantly different
            if lower1 > upper2 or lower2 > upper1:
                different = True
    return num, different     

# Lets test with half of what we have now
num, different = difference_num(50)
print(num, ": ", different) # 50 :  False
num, different = difference_num(25)
print(num, ": ", different) # 25 :  False
num, different = difference_num(13)
print(num, ": ", different) # 13 :  True
num, different = difference_num(19)
print(num, ": ", different) # 19 :  True
num, different = difference_num(22)
print(num, ": ", different) # 22 :  True
num, different = difference_num(24)
print(num, ": ", different) # 24 :  True

# So we need so sample at least 25 data points at every query
