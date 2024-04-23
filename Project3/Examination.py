import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
import math
import os

np.random.seed(42)

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

data6 = pd.read_csv("data_1361.csv") # D = -2
del data6[data6.columns[0]]

# Choose the data to investigate
data = data6

###################################################
## plot all variables against eachother ##
sns.pairplot(data, plot_kws={"s": 5})
plt.show()

# Data1: The relationship between A & F seems quite linear
#        aka with a clear correlation. Furthermore E & F 
#        seems to have a parabolic relationship, and D & E 
#        creates an "H" shape and lastly C & E seems to 
#        create a square.
# Data2/F: D is almost constant at 0 when we intervene with F=0
# Data3/E: No clear relationships
# Data4/C: No clear relationships ish
# Data5/B: Once more D & E creates an "H" shape and E & F 
#        seems to have a parabolic relationship. Furthermore 
#        C and D seems to perhaps be correlated 
# Data6/D: The C & E square is gone 


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
# Data2/F: none
# Data3/E: none
# Data4/C: var A and var F has corr: -0.7077970481544589 
# Data5/B: none
# Data6/D: var A and var F has corr: -0.6580983545598529
#          var B and var C has corr: 0.6450769014297322
#          var B and var F has corr: 0.5375220107260147


###################################################
## Mutual information ##

# Function for calculating mutual information
# and for plotting if wanted
"""
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
"""

def MI(x, y, bandwidth=0.1, plot=False):
    # Convert x and y to np arrays
    x = np.array(x)
    y = np.array(y)
    # Fit KDE to X and Y
    kde_x = KernelDensity(bandwidth=bandwidth).fit(x.reshape(-1, 1))
    kde_y = KernelDensity(bandwidth=bandwidth).fit(y.reshape(-1, 1))
    # Evaluate KDE at sample points
    log_density_x = kde_x.score_samples(x.reshape(-1, 1))
    log_density_y = kde_y.score_samples(y.reshape(-1, 1))
    # Compute joint KDE
    joint_kde = np.exp(log_density_x + log_density_y)
    # Compute marginal KDE
    marginal_kde_x = np.exp(log_density_x)
    marginal_kde_y = np.exp(log_density_y)
    # Compute MI
    eps = np.spacing(1)
    MI = np.sum(joint_kde * np.log((joint_kde + eps) / (marginal_kde_x * marginal_kde_y + eps)))
    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(x, y, alpha=0.2)
        plt.title("Joint Distribution")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.subplot(1, 2, 2)
        plt.scatter(x, marginal_kde_x, label="P(X)", alpha=0.5)
        plt.scatter(y, marginal_kde_y, label="P(Y)", alpha=0.5)
        plt.title("Product of Marginals")
        plt.xlabel("X / Y")
        plt.ylabel("Density")
        plt.legend()
        plt.suptitle(f"Mutual Information: {MI:.4f}")
        plt.show()
    return MI

# create list for storing all MI values
datapoints = []
# Calculate all MI values for all pairs of variables
for row in data:
    for column in data.columns:
        if row != column and MI(data[str(row)], data[str(column)]) not in datapoints:
            datapoints.append(MI(data[str(row)], data[str(column)]))
            print(row, " and ", column, " has MI: ", MI(data[str(row)], data[str(column)]))

# Create boxplot to investigate if there are any MI value outliers
# This would indicate a particually strong/weak relationship between the given variables
plt.boxplot(datapoints, labels = [""])
plt.scatter(np.ones_like(datapoints), datapoints, color='red', marker='o', alpha=0.5)
plt.ylabel("MI values")  
plt.show() 
# Data1:   Positive outlier = B & D
# Data2/F: Outlier = D & F
# Data3/E: Outliers = C & E and D & E
# Data4/C: Outlier = C & F
# Data5/B: Outliers = B & D and B & E
# Data6/D: Positive outliers = D & E and D & F
#          Negative outliers = B & D and B & E and C & F


###################################################
# plotting wobly wobly 3d kind plot shit
#plt.figure()
#sns.jointplot(data={'B':data["A"],'A':data["B"]},x='B',y='A', kind='kde') 
#plt.show()


###################################################
# So what now?

#### After data1:
# We want to sample with the conditioning on F being the highest value:
print(min(data["F"])) # 4.9873 

# After every other sample:
# We want to examine which variables changed distribution
for variable in data:
    plt.boxplot(data1[variable], positions=[1], labels = [variable + "1"])
    plt.boxplot(data[variable], positions=[2], labels = [variable + "2"])
    plt.scatter(np.ones_like(data1[variable]), data1[variable], color='blue', marker='o', alpha=0.5)
    plt.scatter(2 * np.ones_like(data[variable]), data[variable], color='red', marker='x', alpha=0.5)
    plt.ylabel("Values")  
    plt.show()
    
#### After data2/F:
# A clearly changed a lot
# B changed a bit
# C did not change whatsoever
# D changed
# E changed but...

# We now want to sample E (because it has a few possible connections)
# and want to try and use a negative value this time. 
# So we use E=-2, as this is the most extreme we're allowed to




#### After data3/E:
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



#### After data4/C:
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




#### After data5/B:
# A is not really changed
# C is changed a bit
# D is changed a tiny bit
# E is not really changed
# F is changed a tiny bit 



#### After data6/D:
# A changed a bit
# C did not change a lot
# C didn't change
# E didn't change
# F did not change a lot