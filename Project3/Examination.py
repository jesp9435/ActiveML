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
data7 = pd.read_csv("data_1421.csv") # A = 0
del data7[data7.columns[0]]
data8 = pd.read_csv("data_1422.csv") # C = 0 and 50 data points
del data8[data8.columns[0]]
data9 = pd.read_csv("data_1450.csv") # E = 0 and 50 data points
del data9[data9.columns[0]]


# Choose the data to investigate
data = data9

###################################################
## plot all variables against eachother ##
sns.pairplot(data3, plot_kws={"s": 5})
plt.show()

# Data1:   The relationship between A & F seems quite linear
#          aka with a clear correlation. Furthermore E & F 
#          seems to have a parabolic relationship, and D & E 
#          creates an "H" shape and lastly C & E seems to 
#          create a square.
# Data2/F: D is almost constant at 0 when we intervene with F=0
#          The histograms for A and B changed. 
# Data3/E: No clear relationships 
#          Histograms did not change.
# Data4/C: No clear relationships ish
#          Histograms for A, B, D and F changed.
# Data5/B: Once more D & E creates an "H" shape and E & F 
#          seems to have a parabolic relationship. Furthermore 
#          C and D seems to perhaps be correlated 
#          Histogram for F changed a bit.
# Data6/D: The C & E square is gone 
#          Histograms for E and F changed a bit.
# Data7/A: The C & E square is gone 
#          Histogram for F changed
# Data8/C2:Nothing changed...
# Data9/E2:Nothings changed...




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
# Data7/A: var D and var E has corr: -0.5361825807946552      
# Data8/C: none
# Data9/E: var A and var F has corr: -0.5942921880362798 


################################################### 
## Mutual information ##

# Function for calculating mutual information
# and for plotting if wanted
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
# Data7/A: Positive outlier = D & F
# Data8/C2:None
# Data9/E2:Positive outlier = D & E



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
# so let's examine the range: 
print(min(data1["B"])) # -4.522
print(max(data1["B"])) #  4.257
# So we use B=0




#### After data5/B:
# A is not really changed
# C is changed a bit
# D is changed a tiny bit
# E is not really changed
# F is changed a tiny bit 
print(min(data1["D"])) # -1.9571239854278204
print(max(data1["D"])) #  1.619418927458252
# So we use D=-2



#### After data6/D:
# A changed a bit
# B did not change a lot
# C didn't change
# E didn't change
# F did not change a lot
print(min(data1["A"])) # -4.00
print(max(data1["A"])) #  5.55
# So we use A=0



#### After data7/A:
# B didn't change
# C did not change a lot
# D didn't change
# E didn't change
# F changed a tiny bit

# Now we try C=0



#### After data8/C2:
# Nothing changed
# C is a leaf

# Now we try E=0 because we're quite unsure of where it belongs in the graph.


#### After data9/E2:
# D changed dramatically as the only one
# E must point to D and nothing else