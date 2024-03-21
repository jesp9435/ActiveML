import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import modAL
from copy import deepcopy
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
from sklearn.decomposition import PCA

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



method_performances = []


for i in range(2): #loop through our 3 methods
    query_members_performance_history = []
    for n in [5]: #[5, 10, 15]                           ################### remember to change this ############################
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
                else:
                    query_idx = consensus_disagreement(predictions)

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
            
        query_members_performance_history.append(performance_history)
    
    method_performances.append(query_members_performance_history)

# Plot our performance over time. ########## FIX TO LOOK LIKE EXAMPLE REPORT ###########
fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)

ax.plot(performance_history)
ax.scatter(range(len(performance_history)), performance_history, s=13)

ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=10))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))

ax.set_ylim(bottom=0, top=1)
ax.grid(True)

ax.set_title('Incremental classification accuracy')
ax.set_xlabel('Query iteration')
ax.set_ylabel('Classification Accuracy')

plt.show()


## plot
plt.plot(range(len(method_performances[0][0])), method_performances[0][0], label = "Vote entropy 5 members" , color = "red")
plt.plot(range(len(method_performances[0][1])), method_performances[0][1], label = "Vote entropy 10 members" , color = "blue")
plt.plot(range(len(method_performances[0][2])), method_performances[0][2], label = "Vote entropy 15 members" , color = "green")
plt.legend()
plt.show()
