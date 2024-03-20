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

"""
# Visualizing the data
_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()
"""

""" # Maybe just delete? 
# visualizing the classes
with plt.style.context('seaborn-v0_8-white'):
    plt.figure(figsize=(7, 7))
    pca = PCA(n_components=2).fit_transform(iris['data'])
    plt.scatter(x=pca[:, 0], y=pca[:, 1], c=iris['target'], cmap='viridis', s=50)
    plt.title('The iris dataset')
    plt.show()
""" 
pca = PCA(n_components=2).fit_transform(iris['data']) #why is n_components 2?

# generate the pool
X_pool = deepcopy(iris['data'])
y_pool = deepcopy(iris['target']) # label



#############################################
# A way to calculate vote entropy?:
#modAL.disagreement.vote_entropy()
#############################################

# committee = Committee(learner_list=ActiveLearner)


def KLD(X_pool, committee):
    # Calculate the predictions of the committee members
    predictions = committee.predict_proba(X_pool)

    # Calculate the consensus (mean) prediction
    consensus_prediction = predictions.mean(axis=0)

    # Calculate the KL divergence for each data point
    KL_divergence = np.sum(predictions * np.log(predictions / consensus_prediction + 1e-10), axis=1).mean(axis=0)

    # Select the data point with the maximum KL divergence
    query_idx = np.argmax(KL_divergence)
    return query_idx 

query_members_performance_history = []

    
for n in [5]: #[5, 10, 15] 
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
    
    """
    # Visualizing how the models predict after inital training
    with plt.style.context('seaborn-v0_8-white'):
        plt.figure(figsize=(n_members*7, 7))
        for learner_idx, learner in enumerate(committee):
            plt.subplot(1, n_members, learner_idx + 1)
            plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(iris['data']), cmap='viridis', s=50)
            plt.title('Learner no. %d initial predictions' % (learner_idx + 1))
        plt.show()
    """

    # Calculating the mean accuracy of the committee so far
    unqueried_score = committee.score(iris['data'], iris['target'])

    """
    # Visualizing how the committee predict given inital training
    with plt.style.context('seaborn-v0_8-white'):
        plt.figure(figsize=(7, 7))
        prediction = committee.predict(iris['data'])
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
        plt.title('Committee initial predictions, accuracy = %1.3f' % unqueried_score)
        plt.show()
    """

    # Create a list to store committee accuracy/performance over time
    performance_history = [unqueried_score]

    # Query by committee 
    n_queries = 25 # Number of queries in total (so we end up using 127/150 data points)
    for idx in range(n_queries):
        for n in range(5): # Query 5 data points at a time
            
            query_idx = KLD(X_pool, committee)

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



    """
    # Visualizing the final predictions per learner
    with plt.style.context('seaborn-v0_8-white'):
        plt.figure(figsize=(n_members*7, 7))
        for learner_idx, learner in enumerate(committee):
            plt.subplot(1, n_members, learner_idx + 1)
            plt.scatter(x=pca[:, 0], y=pca[:, 1], c=learner.predict(iris['data']), cmap='viridis', s=50)
            plt.title('Learner no. %d predictions after %d queries' % (learner_idx + 1, n_queries))
        plt.show()

    # Visualizing the Committee's predictions
    with plt.style.context('seaborn-v0_8-white'):
        plt.figure(figsize=(7, 7))
        prediction = committee.predict(iris['data'])
        plt.scatter(x=pca[:, 0], y=pca[:, 1], c=prediction, cmap='viridis', s=50)
        plt.title('Committee predictions after %d queries, accuracy = %1.3f'
                % (n_queries, committee.score(iris['data'], iris['target'])))
        plt.show()
    """



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

