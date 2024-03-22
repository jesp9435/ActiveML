import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, entropy
import pickle 


# Load total_acc from the file
with open('/path/to/your/directory/total_acc.pkl', 'rb') as f:
    total_acc = pickle.load(f)



### Plot our performance over time. 

# initialize a list for storing average and 95% confidence interval
average_data = [[],[],[]] # on the form [members5, members10, members15]

# loop over the committee sizes
for n in range(3): # loop over number of committee sizes (5, 10, 15)
    datalist = total_acc[n] # determine the list to extract data from
    appendlist = average_data[n] # determine the list to append to
    
    for i in range(4): # loop over number of methods (incl baseline)
        # Calculate the mean and standard error for each method
        method_data = [row[i] for row in datalist]
        mean_method_data = np.mean(method_data, axis=0)
        #mean_method_data = [num+20 for num in mean_method_data if num > 50 and num < 80]
        sem_method_data = sem(method_data, axis=0)
        appendlist.append([mean_method_data, sem_method_data])
    
# Create x values
x = np.arange(1, len(average_data[0][0][0])+1)

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
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
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
plt.ylim(0,1)
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
plt.ylim(0,1)
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
plt.ylim(0,1)
plt.ylabel("Accuracy in %")
plt.title("Best vs baseline")
plt.legend()
plt.grid(True)
plt.show()





#####################################################
# acc vs number of members

plt.figure(figsize=(10, 6))
# Vote entropy 5 members
plt.plot(x, average_data[0][0][0], label="5 members KLD", marker='o') #average_data[0][1][0] = [5 members][KLD][mean]
plt.fill_between(x, average_data[0][0][0] - 1.96 * average_data[0][0][1], average_data[0][0][0] + 1.96 * average_data[0][0][1], alpha=0.2)
# Vote entropy 10 members
plt.plot(x, average_data[1][0][0], label="10 members vote entropy", marker='o') 
plt.fill_between(x, average_data[1][0][0] - 1.96 * average_data[1][0][1], average_data[1][0][0] + 1.96 * average_data[1][0][1], alpha=0.2)
# Vote entropy 15 members
plt.plot(x, average_data[2][0][0], label="15 members consensus prob", marker='o') 
plt.fill_between(x, average_data[2][0][0] - 1.96 * average_data[2][0][1], average_data[2][0][0] + 1.96 * average_data[2][0][1], alpha=0.2)

plt.xlabel("Query ID")
plt.ylim(0,1)
plt.ylabel("Accuracy in %")
plt.title("Vote entropy")
plt.legend()
plt.grid(True)
plt.show()