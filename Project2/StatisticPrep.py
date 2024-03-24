import pickle 


# Load total_acc from the other file
with open('C:/Users/cheli/Downloads/total_acc.pkl', 'rb') as f:
    total_acc = pickle.load(f)
    
    
# right now we have:

# total_acc = [members5_acc, members10_acc, members15_acc]
# members5_acc = [[vote_entropy_acc, KLD_acc, cons_acc, baseline_acc], [...]]
    # len(members5_acc) = 10
# members5_acc[0] = [vote_entropy_acc, KLD_acc, cons_acc, baseline_acc]
# baseline_acc = [0.4, 0.67, ...]
    # len(baseline_acc) = 20
    
    
# And we want:

all_acc = [["nameone", "nametwo", "namethree"], 
           [1,2,3], 
           [4,5,6],
           [7,8,9]]

import csv

# Specify the filename for the CSV
filename = "C:/Users/cheli/Downloads/all_acc.csv"

# Write the data to the CSV file
with open(filename, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(all_acc)

print(f"Data exported to {filename}")


