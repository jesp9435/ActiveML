import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score

# Load data
data = pd.read_csv("C:/Users/cheli/Downloads/data_1200.csv")
del data[data.columns[0]] # Remove column of row number


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
plt.boxplot(datapoints)
plt.show()
# from the plotted boxplot we do not get any outliers e.i., MI tells us nothing


###################################################
# plotting wobly wobly 3d kind plot shit
plt.figure()
sns.jointplot(data={'B':data["A"],'A':data["B"]},x='B',y='A', kind='kde') 
plt.show()


###################################################
