import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import modAL
from copy import deepcopy
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
from sklearn.decomposition import PCA
import random
import seaborn as sns
from scipy.stats import sem


# Load the data set
iris = datasets.load_iris() 
np.random.seed(32)

pca = PCA(n_components=2).fit_transform(iris['data']) #why is n_components 2?

# generate the pool
X_pool = deepcopy(iris['data'])
y_pool = deepcopy(iris['target']) # label



# def 1
def vote_entropy(predictions):
    # Calculate the vote proportions for each class
    vote_proportions = np.mean(predictions > 0.5, axis=0)

    # Calculate the vote entropy for each data point
    vote_entropy = entropy(vote_proportions.T + 1e-10, base=2, axis=1)

    # Select the data point with the maximum vote entropy
    query_idx = np.argmax(vote_entropy)
    return query_idx 

# def 2
def KLD(predictions):
    # Calculate the consensus (mean) prediction
    consensus_prediction = predictions.mean(axis=0)

    # Calculate the KL divergence for each data point
    KL_divergence = np.sum(predictions * np.log(predictions / consensus_prediction + 1e-10), axis=1).mean(axis=0)

    # Select the data point with the maximum KL divergence
    query_idx = np.argmax(KL_divergence)
    return query_idx 

# def 3
def consensus_disagreement(predictions):
    # Calculate the consensus (mean) prediction
    consensus_prediction = predictions.mean(axis=0)

    # Calculate the consensus disagreement for each data point
    consensus_disagreement = -np.abs(0.5 - consensus_prediction[:, 0])

    # Select the data point with the maximum consensus disagreement
    query_idx = np.argmax(consensus_disagreement)
    return query_idx 

def baseline(X_pool):
    query_idx = random.randint(0, len(X_pool))
    return query_idx



# Initialize lists for storing accuracies for different committee sizes
members5_acc = [] # on the form [[vote_entropy_acc, KLD_acc, cons_acc, baseline_acc], [...]]
members10_acc = []
members15_acc = []


#repeat the whole experiment 10 times to achieve confidence intervals
for repetition in range(10): 
    # loop through the number of committee members
    for n in [5, 10, 15]:
        # Create a list to store accuracies within each method
        member_acc_across_methods = [] # on the form [vote_entropy_acc, KLD_acc, cons_acc, baseline_acc]
        #loop through our 3 methods and the baseline   
        for i in range(3):                        
            # initializing Committee members
            n_members = n
            learner_list = list()

            # Loop over members of the committee
            for member_idx in range(n_members):
                # initial training data
                n_initial = 2 # number of random data points for inital training
                train_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)
                X_train = X_pool[train_idx]
                y_train = y_pool[train_idx]

                # creating a reduced copy of the data with the known instances removed
                X_pool = np.delete(X_pool, train_idx, axis=0)
                y_pool = np.delete(y_pool, train_idx)

                # initializing learner
                learner = ActiveLearner(
                    estimator=RandomForestClassifier(),
                    X_training=X_train, y_training=y_train
                )
                learner_list.append(learner)

            # assembling the committee
            committee = Committee(learner_list=learner_list)
            
            # Calculating the mean accuracy of the committee so far
            unqueried_score = committee.score(iris['data'], iris['target'])

            # Create a list to store committee accuracy/performance over time
            performance_history = [unqueried_score]

            # Query by committee 
            n_queries = 25 # Number of queries in total (so we end up using 127/150 data points)
            for idx in range(n_queries):
                for n in range(5): # Query 5 data points at a time
                    # Calculate the predictions of the committee members
                    predictions = committee.predict_proba(X_pool)

                    if i == 0:
                        query_idx = vote_entropy(predictions)
                    elif i == 1:
                        query_idx = KLD(predictions)
                    elif i == 2:
                        query_idx = consensus_disagreement(predictions)
                    else:
                        query_idx = baseline(X_pool)

                    #query_idx, query_instance = committee.query(X_pool) 
                    committee.teach(
                        X=X_pool[query_idx].reshape(1, -1),
                        y=y_pool[query_idx].reshape(1, )
                    )

                    # Remove queried instance from pool
                    X_pool = np.delete(X_pool, query_idx, axis=0)
                    y_pool = np.delete(y_pool, query_idx)

                # Calculate performance after query and add to performance history
                performance_history.append(committee.score(iris['data'], iris['target']))
                
            member_acc_across_methods.append(performance_history)
        
        if n == 5:
            members5_acc.append(member_acc_across_methods)
        elif n == 10:
            members10_acc.append(member_acc_across_methods)
        else:
            members15_acc.append(member_acc_across_methods)




total_acc = [members5_acc, members10_acc, members15_acc]



### Plot our performance over time. 

# initialize a list for storing average and 95% confidence interval
average_data = [[],[],[]] # on the form [members5, members10, members15]

# loop over the committee sizes
for n in range(3): # loop over number of committee sizes (5, 10, 15)
    datalist = total_acc[n] # determine the list to extract data from
    appendlist = average_data[n] # determine the list to append to
    
    for i in range(4):
        # Calculate the mean and standard error for each method
        method_data = [row[i] for row in datalist]
        mean_method_data = np.mean(method_data, axis=0)
        #mean_method_data = [num+20 for num in mean_method_data if num > 50 and num < 80]
        sem_method_data = sem(method_data, axis=0)
        appendlist.append([mean_method_data, sem_method_data])
    
# Create x values
x = np.arange(1, 11)

for i in range(3):# loop over number of committee sizes (5, 10, 15)
    member_num = average_data[i]
    # Create the plot
    plt.figure(figsize=(10, 6))
    for method in range(len(member_num)):
        if method == 0:
            plt.plot(x, member_num[method][0], label="Vote Entropy", marker='o')
        elif method == 1:
            plt.plot(x, member_num[method][0], label="KLD", marker='o')
        elif method == 2:
            plt.plot(x, member_num[method][0], label="Consensus", marker='o')
        else:
            plt.plot(x, member_num[method][0], label="Baseline", marker='o')
        plt.fill_between(x, member_num[method][0] - 1.96 * member_num[method][1], member_num[method][0] + 1.96 * member_num[method][1], alpha=0.2)

    plt.xlabel("Query ID")
    plt.ylim(0,100)
    plt.ylabel("Accuracy in %")
    plt.title("Committee size " + str([5, 10, 15][i]))
    plt.legend()
    plt.grid(True)
    plt.show()


### plot the best performing against eachother (only for us)
plt.figure(figsize=(10, 6))
# Best of 5 members
plt.plot(x, average_data[0][1][0], label="5 members KLD", marker='o') #average_data[0][1][0] = [5 members][KLD][mean]
plt.fill_between(x, average_data[0][1][0] - 1.96 * average_data[0][1][1], average_data[0][1][0] + 1.96 * average_data[0][1][1], alpha=0.2)
# Best of 10 members
plt.plot(x, average_data[1][0][0], label="10 members vote entropy", marker='o') 
plt.fill_between(x, average_data[1][0][0] - 1.96 * average_data[1][0][1], average_data[1][0][0] + 1.96 * average_data[1][0][1], alpha=0.2)
# Best of 15 members
plt.plot(x, average_data[2][2][0], label="15 members consensus prob", marker='o') 
plt.fill_between(x, average_data[2][2][0] - 1.96 * average_data[2][2][1], average_data[2][2][0] + 1.96 * average_data[2][2][1], alpha=0.2)

plt.xlabel("Query ID")
plt.ylim(0,100)
plt.ylabel("Accuracy in %")
plt.title("The best performing")
plt.legend()
plt.grid(True)
plt.show()


### plot the baselines against eachother (only for us)
plt.figure(figsize=(10, 6))
# Best of 5 members
plt.plot(x, average_data[0][3][0], label="5 members", marker='o') #average_data[0][1][0] = [5 members][KLD][mean]
plt.fill_between(x, average_data[0][3][0] - 1.96 * average_data[0][3][1], average_data[0][3][0] + 1.96 * average_data[0][3][1], alpha=0.2)
# Best of 10 members
plt.plot(x, average_data[1][3][0], label="10 members", marker='o') 
plt.fill_between(x, average_data[1][3][0] - 1.96 * average_data[1][3][1], average_data[1][3][0] + 1.96 * average_data[1][3][1], alpha=0.2)
# Best of 15 members
plt.plot(x, average_data[2][3][0], label="15 members", marker='o') 
plt.fill_between(x, average_data[2][3][0] - 1.96 * average_data[2][3][1], average_data[2][3][0] + 1.96 * average_data[2][3][1], alpha=0.2)

plt.xlabel("Query ID")
plt.ylim(0,100)
plt.ylabel("Accuracy in %")
plt.title("Baselines")
plt.legend()
plt.grid(True)
plt.show()



### plot the best combo against the best baseline
plt.figure(figsize=(10, 6))
# Best combo
plt.plot(x, average_data[0][1][0], label="5 members KLD", marker='o') 
plt.fill_between(x, average_data[0][1][0] - 1.96 * average_data[0][1][1], average_data[0][1][0] + 1.96 * average_data[0][1][1], alpha=0.2)
# best performing baseline
plt.plot(x, average_data[1][3][0], label="10 members baseline", marker='o') 
plt.fill_between(x, average_data[1][3][0] - 1.96 * average_data[1][3][1], average_data[1][3][0] + 1.96 * average_data[1][3][1], alpha=0.2)
plt.xlabel("Query ID")
plt.ylim(0,100)
plt.ylabel("Accuracy in %")
plt.title("Best vs baseline")
plt.legend()
plt.grid(True)
plt.show()