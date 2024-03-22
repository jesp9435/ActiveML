import numpy as np
import matplotlib.pyplot as plt
import modAL
from copy import deepcopy
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner, Committee
import random
from scipy.stats import sem, entropy
import pickle 

np.random.seed(32) 


# def 1
def vote_entropy(predictions):
    # Calculate the vote proportions for each class
    vote_proportions = np.mean(predictions > 0.5, axis=0)

    # Calculate the vote entropy for each data point
    vote_entropy = entropy(vote_proportions.T + 1e-10, base=2, axis=0)

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
    consensus_disagreement = -np.abs(0.5 - consensus_prediction)

    # Select the data point with the maximum consensus disagreement
    query_idx = np.argmax(consensus_disagreement)
    return query_idx 

def baseline(X_pool):
    query_idx = random.randint(0, len(X_pool)-1)
    return query_idx



# Initialize lists for storing accuracies for different committee sizes
members5_acc = [] # on the form [[vote_entropy_acc, KLD_acc, cons_acc, baseline_acc], [...]]
members10_acc = []
members15_acc = []


# repeat the whole experiment 10 times to achieve confidence intervals
for repetition in range(10): 
    print("Repetition:", repetition+1, "/5")
    # loop through the number of committee members
    for n in [5, 10, 15]:
        print("Number of members:", n)
        # Create a list to store accuracies within each method
        member_acc_across_methods = [] # on the form [vote_entropy_acc, KLD_acc, cons_acc, baseline_acc]
        #loop through our 3 methods and the baseline   
        for i in range(4):   
            print("Method:", i+1, "/4")
            # Load the data set
            iris = datasets.load_iris() 
            
            # generate the pool
            X_pool = deepcopy(iris['data'])
            y_pool = deepcopy(iris['target']) # label

            # initializing Committee members
            n_members = n
            learner_list = list()

            # Loop over members of the committee
            for member_idx in range(n_members):
                print("Member:", member_idx+1, "/", n_members)
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
            n_queries = 20 # Number of queries in total (so we end up using 120/150 data points)
            for idx in range(n_queries):
                print("idx:", idx+1, "/20")
                for _ in range(5): # Query 5 data points at a time
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

with open('/path/to/your/directory/total_acc.pkl', 'wb') as f:
    pickle.dump(total_acc, f) 













