import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("/Users/jesperberglund/Downloads/data_1200.csv")
del data[data.columns[0]]
print(type(data))
# sammenhæng mellem variable parret to-og-to
sns.pairplot(data)
#plt.show()

# creation of correlation matrix
corrM = data.corr()
print(corrM)

# vi vil kun have vist, hvor corr >= 0.5
for row in corrM:
    for column in corrM.columns:
        value = corrM[row][column]
        if abs(value) >= 0.5 and value < 1:
            print("var",row," and var",column, "has corr: ",value)


## check for mutual information ##
def MI(x,y,Nbins=21):
    bins = np.linspace(np.min(x),np.max(x),Nbins)
    eps=np.spacing(1)
    x_marginal = np.histogram(x,bins=bins)[0]
    x_marginal = x_marginal/x_marginal.sum()
    y_marginal = np.array(np.histogram(y,bins=bins)[0])
    y_marginal = y_marginal/y_marginal.sum()
    xy_joint = np.array(np.histogram2d(x,y,bins=(bins,bins))[0])
    xy_joint = xy_joint/xy_joint.sum()
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(xy_joint.T,origin='lower')
    plt.title('joint')
    plt.subplot(1,2,2)
    plt.imshow((x_marginal[:,None]*y_marginal[None,:]).T,origin='lower')
    plt.title('product of marginals')
    MI=np.sum(xy_joint*np.log(xy_joint/(x_marginal[:,None]*y_marginal[None,:]+eps)+eps))
    plt.suptitle('Mutual information: %f'%MI)
    #plt.show()
    return(MI)

for row in data:
    for column in data.columns:
        if MI(data[str(row)], data[str(column)]) >= 0.1:
            print(row, " and ", column, " has MI: ", MI(data[str(row)], data[str(column)]))


plt.figure()
sns.jointplot(data={'x':x,'y':y},x='x',y='y',kind='kde')
plt.show()