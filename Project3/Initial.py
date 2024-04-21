import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats


data1 = pd.read_csv("/Users/jesperberglund/Downloads/data_1200.csv")
del data1[data1.columns[0]]
data2 = pd.read_csv("/Users/jesperberglund/Downloads/data_1308.csv")
del data2[data2.columns[0]]

data = data2

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

# var A and var F has corr: -0.5407078849279002


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
            #print(row, " and ", column, " has MI: ", MI(data[str(row)], data[str(column)]))

# Create boxplot to investigate if there are any MI value outliers
# This would indicate a particually strong/weak relationship between the given variables
plt.boxplot(datapoints, labels = [""])
plt.ylabel("MI values")  
plt.show()
# from the plotted boxplot we do not get any outliers e.i., MI tells us nothing


###################################################
# plotting wobly wobly 3d kind plot shit
plt.figure()
sns.jointplot(data={'B':data["A"],'A':data["B"]},x='B',y='A', kind='kde') 
plt.show()


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



# We want to sample with the conditioning on F being the highest value:
print(max(data["F"])) # 4.9873 = 5




