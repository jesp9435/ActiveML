import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, entropy
import pickle 


# Load total_acc from the other file
with open('C:/Users/cheli/Downloads/total_acc.pkl', 'rb') as f:
    total_acc = pickle.load(f)


### Create average data with 95% confidence bands ###

# initialize a list for storing average and 95% confidence interval
average_data = [[],[],[]] # on the form [members5, members10, members15]

# loop over the committee sizes
for size in range(3): # loop over number of committee sizes (5, 10, 15)
    # determine the list to extract data from
    datalist = total_acc[size] # from total_acc = [members5_acc, members10_acc, members15_acc]
    # determine the list to append to
    appendlist = average_data[size] # from average_data = [[],[],[]]
    
    for method in range(3): # loop over number of methods (excl baseline)
        # Calculate the mean and standard error for each method
        method_data = [row[method] for row in datalist]
        mean_method_data = np.mean(method_data, axis=0)
        sem_method_data = sem(method_data, axis=0)
        appendlist.append([mean_method_data, sem_method_data])


### Create only one average graph for baseline across all committee sizes

# Store all basline data in a matrix
all_baseline_data = []
for n_members in total_acc: # loop through [members5_acc, members10_acc, members15_acc]
    for repetition in n_members: # loop through [[vote_entropy_acc, KLD_acc, cons_acc, baseline_acc], [...]]
        # append each graph data so baseline_data = [[0.4, 0.67, ...], [...]]
        all_baseline_data.append(repetition[3]) 

# Calculate the average 
baseline_data = []
mean_data = np.mean(all_baseline_data, axis=0)
sem_data = sem(all_baseline_data, axis=0)
baseline_data.append([mean_data, sem_data])
baseline_data = baseline_data[0]


        

### Plot performance over time ###

# Create x values
x = np.arange(1, len(average_data[0][0][0])+1)


### Plot graphs for each method within each committee size (only for us)
for i in range(3):# loop over number of committee sizes (5, 10, 15)
    member_num = average_data[i]
    # Create the plot
    plt.figure(figsize=(10, 6))
    for method in range(len(member_num)): # loop over methods
        if method == 0:
            plt.plot(x, member_num[method][0], label="Vote Entropy", marker='o')
        elif method == 1:
            plt.plot(x, member_num[method][0], label="KLD", marker='o')
        elif method == 2:
            plt.plot(x, member_num[method][0], label="Consensus", marker='o')
        else:
            plt.plot(x, member_num[method][0], label="Baseline", marker='o')
        plt.fill_between(x, member_num[method][0] - 1.96 * member_num[method][1], member_num[method][0] + 1.96 * member_num[method][1], alpha=0.2)

    plt.plot(x, baseline_data[0], label="Baseline", marker='o') 
    plt.fill_between(x, baseline_data[0] - 1.96 * baseline_data[1], baseline_data[0] + 1.96 * baseline_data[1], alpha=0.2)
    plt.xlabel("Query ID")
    plt.ylim(0,1)
    plt.xlim(1,21)
    plt.xticks(range(1,22))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.ylabel("Accuracy")
    plt.title("Committee size " + str([5, 10, 15][i]))
    plt.legend()
    plt.grid(True)
    plt.show()



### Plot graphs for each method with committee size = 10
member_num = average_data[1]
# Create the plot
plt.figure(figsize=(10, 6))
for method in range(len(member_num)): # loop over methods
    if method == 0:
        plt.plot(x, member_num[method][0], label="Vote Entropy", marker='o')
    elif method == 1:
        plt.plot(x, member_num[method][0], label="KLD", marker='o')
    elif method == 2:
        plt.plot(x, member_num[method][0], label="Consensus", marker='o')
    else:
        plt.plot(x, member_num[method][0], label="Baseline", marker='o')
    plt.fill_between(x, member_num[method][0] - 1.96 * member_num[method][1], member_num[method][0] + 1.96 * member_num[method][1], alpha=0.2)

plt.plot(x, baseline_data[0], label="Baseline", marker='o') 
plt.fill_between(x, baseline_data[0] - 1.96 * baseline_data[1], baseline_data[0] + 1.96 * baseline_data[1], alpha=0.2)
plt.xlabel("Query ID")
plt.ylim(0,1)
plt.xlim(1,21)
plt.xticks(range(1,22))
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel("Accuracy")
plt.title("Committee size 10")
plt.legend()
plt.grid(True)
plt.show()




### Plot graphs for each member size within each method (only for us)
for method in range(3):
    plt.figure(figsize=(10, 6))
    # Vote entropy 5 members
    plt.plot(x, average_data[0][method][0], label="5 members", marker='o') #average_data[0][1][0] = [5 members][KLD][mean]
    plt.fill_between(x, average_data[0][method][0] - 1.96 * average_data[0][method][1], average_data[0][method][0] + 1.96 * average_data[0][method][1], alpha=0.2)
    # Vote entropy 10 members
    plt.plot(x, average_data[1][method][0], label="10 members", marker='o') 
    plt.fill_between(x, average_data[1][method][0] - 1.96 * average_data[1][method][1], average_data[1][method][0] + 1.96 * average_data[1][method][1], alpha=0.2)
    # Vote entropy 15 members
    plt.plot(x, average_data[2][method][0], label="15 members", marker='o') 
    plt.fill_between(x, average_data[2][method][0] - 1.96 * average_data[2][method][1], average_data[2][method][0] + 1.96 * average_data[2][method][1], alpha=0.2)
    # Baseline
    plt.plot(x, baseline_data[0], label="Baseline", marker='o') 
    plt.fill_between(x, baseline_data[0] - 1.96 * baseline_data[1], baseline_data[0] + 1.96 * baseline_data[1], alpha=0.2)
        
    plt.xlabel("Query ID")
    plt.ylim(0,1)
    plt.xticks(range(1,22))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.ylabel("Accuracy")
    plt.title("Method: " + ["Vote entropy", "KLD", "Consensus prob"][method])
    plt.legend()
    plt.grid(True)
    plt.show()


### Plot graphs for each member size with method = vote entropy
method = 0
plt.figure(figsize=(10, 6))
# Vote entropy 5 members
plt.plot(x, average_data[0][method][0], label="5 members", marker='o') #average_data[0][1][0] = [5 members][KLD][mean]
plt.fill_between(x, average_data[0][method][0] - 1.96 * average_data[0][method][1], average_data[0][method][0] + 1.96 * average_data[0][method][1], alpha=0.2)
# Vote entropy 10 members
plt.plot(x, average_data[1][method][0], label="10 members", marker='o') 
plt.fill_between(x, average_data[1][method][0] - 1.96 * average_data[1][method][1], average_data[1][method][0] + 1.96 * average_data[1][method][1], alpha=0.2)
# Vote entropy 15 members
plt.plot(x, average_data[2][method][0], label="15 members", marker='o') 
plt.fill_between(x, average_data[2][method][0] - 1.96 * average_data[2][method][1], average_data[2][method][0] + 1.96 * average_data[2][method][1], alpha=0.2)
# Baseline
plt.plot(x, baseline_data[0], label="Baseline", marker='o') 
plt.fill_between(x, baseline_data[0] - 1.96 * baseline_data[1], baseline_data[0] + 1.96 * baseline_data[1], alpha=0.2)
    
plt.xlabel("Query ID")
plt.ylim(0,1)
plt.xticks(range(1,22))
plt.yticks(np.arange(0,1.1,0.1))
plt.ylabel("Accuracy")
plt.title("Disagreement metric: Vote entropy")
plt.legend()
plt.grid(True)
plt.show()



