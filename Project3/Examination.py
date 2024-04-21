import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
import os

os.chdir("C:/Users/cheli/Downloads")

data1 = pd.read_csv("data_1200.csv")
del data1[data1.columns[0]]

data2 = pd.read_csv("data_1308.csv") # F = 2
del data2[data2.columns[0]]

data3 = pd.read_csv("data_1309.csv") # E = -2
del data3[data3.columns[0]]

data4 = pd.read_csv("data_1311.csv") # C = -2
del data4[data4.columns[0]]

data5 = pd.read_csv("data_1312.csv") # B = 0
del data5[data5.columns[0]]

data = data5
# plot all variables against eachother
sns.pairplot(data, plot_kws={"s": 5})
plt.show()


###################################################
## Correlation ##

# create a correlation matrix
corrM = data.corr()

# Inform us of which pairs has corr >= 0.5
for row in corrM:
    for column in corrM.columns:
        value = corrM[row][column]
        if abs(value) >= 0.5 and value < 1:
            print("var",row,"and var",column, "has corr:",value)

# Data1: var A and var F has corr: -0.5407078849279002
# Data2: none
# Data3: none
# Data4: var A and var F has corr: -0.7077970481544589
# Data3: none



###################################################
## Mutual information ##

# Function for calculating mutual information
# and for plotting if wanted
def MI(x,y,Nbins=21, plot = False):
    bins = np.linspace(np.min(x),np.max(x),Nbins)
    eps=np.spacing(1)
    x_marginal = np.histogram(x,bins=bins)[0]
    x_marginal = x_marginal/x_marginal.sum()
    y_marginal = np.array(np.histogram(y,bins=bins)[0])
    y_marginal = y_marginal/y_marginal.sum()
    xy_joint = np.array(np.histogram2d(x,y,bins=(bins,bins))[0])
    xy_joint = xy_joint/xy_joint.sum()
    MI=np.sum(xy_joint*np.log(xy_joint/(x_marginal[:,None]*y_marginal[None,:]+eps)+eps))
    if plot:
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(xy_joint.T,origin='lower')
        plt.title('joint')
        plt.subplot(1,2,2)
        plt.imshow((x_marginal[:,None]*y_marginal[None,:]).T,origin='lower')
        plt.title('product of marginals')
        plt.suptitle('Mutual information: %f'%MI)
        plt.show()
    return(MI)


# create list for storing all MI values
datapoints = []
# Calculate all MI values for all pairs of variables
for row in data:
    for column in data.columns:
        if row != column:
            datapoints.append(MI(data[str(row)], data[str(column)]))
            print(row, " and ", column, " has MI: ", MI(data[str(row)], data[str(column)]))

# Make sure there is no nan values by replacing them with 0
datapoints = [n for n in datapoints if not isinstance(n, float) or not math.isnan(n)]

# Create boxplot to investigate if there are any MI value outliers
# This would indicate a particually strong/weak relationship between the given variables
plt.boxplot(datapoints, labels = [""])
plt.scatter(np.ones_like(datapoints), datapoints, color='red', marker='o', alpha=0.5)
plt.ylabel("MI values")  
plt.show()
#Data1: from the plotted boxplot we do not get any outliers e.i., MI tells us nothing
#Data2: There is one outlier of an encredible low value between C and F 
#Data3: There is one outlier of an encredible low value between A and E
#Data4: There are two outliers: A and C as well as B and C
#Data5: No outliers, but low: C & B, D & B, E & B, F & B 


###################################################
# plotting wobly wobly 3d kind plot shit
#plt.figure()
#sns.jointplot(data={'B':data["A"],'A':data["B"]},x='B',y='A', kind='kde') 
#plt.show()


###################################################
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



###################################################
# So what now?

#### After data1:
# We want to sample with the conditioning on F being the highest value:
print(min(data["F"])) # 4.9873 

#### After data2:
# We want to examine which variables changed distribution
for variable in data:
    plt.boxplot(data1[variable], positions=[1], labels = [variable + "1"])
    plt.boxplot(data2[variable], positions=[2], labels = [variable + "2"])
    plt.scatter(np.ones_like(data1[variable]), data1[variable], color='blue', marker='o', alpha=0.5)
    plt.scatter(2 * np.ones_like(data2[variable]), data2[variable], color='red', marker='x', alpha=0.5)
    plt.ylabel("Values")  
    plt.show()
# A clearly changed a lot
# B changed a bit
# C did not change whatsoever
# D changed
# E changed but...

# We now want to sample E (because we don't know a lot about it)
# and want to try and use a negative value this time. 
# So we use E=-2, as this is the most extreme we're allowed to



#### After data3:
# We want to examine which variables changed distribution
for variable in data:
    plt.boxplot(data1[variable], positions=[1], labels = [variable + "1"])
    plt.boxplot(data3[variable], positions=[2], labels = [variable + "2"])
    plt.scatter(np.ones_like(data1[variable]), data1[variable], color='blue', marker='o', alpha=0.5)
    plt.scatter(2 * np.ones_like(data3[variable]), data3[variable], color='red', marker='x', alpha=0.5)
    plt.ylabel("Values")  
    plt.show()
# A changed a lot
# B changed 
# C did not change whatsoever
# D changed a bit but...
# F changed a lot 

# We now want to sample C (because we don't know a lot about it)
# and want to try and use an extreme value. 
print(min(data1["C"])) # -1.459 
print(max(data1["C"])) #  3.568
# So we use C=-2



#### After data4:
# We want to examine which variables changed distribution
for variable in data:
    plt.boxplot(data1[variable], positions=[1], labels = [variable + "1"])
    plt.boxplot(data4[variable], positions=[2], labels = [variable + "2"])
    plt.scatter(np.ones_like(data1[variable]), data1[variable], color='blue', marker='o', alpha=0.5)
    plt.scatter(2 * np.ones_like(data4[variable]), data4[variable], color='red', marker='x', alpha=0.5)
    plt.ylabel("Values")  
    plt.show()
# A did not change a lot
# B changed a tiny bit
# D did not change whatsoever
# E did not change a lot
# F changed a tiny bit

# We now want to sample B (because we don't know a lot about it)
# and want to try and use an extreme value. 
print(min(data1["B"])) # -4.522
print(max(data1["B"])) #  4.257
# So we use B=0




#### After data5:
# We want to examine which variables changed distribution
for variable in data:
    plt.boxplot(data1[variable], positions=[1], labels = [variable + "1"])
    plt.boxplot(data5[variable], positions=[2], labels = [variable + "2"])
    plt.scatter(np.ones_like(data1[variable]), data1[variable], color='blue', marker='o', alpha=0.5)
    plt.scatter(2 * np.ones_like(data5[variable]), data5[variable], color='red', marker='x', alpha=0.5)
    plt.ylabel("Values")  
    plt.show()
# A is not really changed
# C is changed a bit
# D is changed a tiny bit
# E is not really changed
# F is changed a tiny bit 