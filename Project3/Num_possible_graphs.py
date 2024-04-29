import math

# Python implementation of the formula from page 159 of "Causality.pdf"

def DAG(N: int) -> int:
    '''
    Parameters: N (int) - number of nodes in the Directed Acyclic Graph (DAG) \n
    Return the number of Directed Acyclic Graphs (DAGs) with N nodes.
    '''
    count = 0
    # Base case 
    if N == 0:
        return 1
    # Recursive case
    for k in range(1, N+1):
        count += ((-1)**(k-1)) * math.comb(N, k) * 2**(k*(N-k)) * DAG(N - k)
    return count

print(DAG(7)) # 1138779265