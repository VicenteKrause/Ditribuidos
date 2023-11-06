# My Utility : 

import numpy  as np 
import pandas as pd

# estimation of information
def information_gain(Y, x):
    Y = pd.DataFrame(Y)
    class_counts = Y.value_counts()
    class_entropy = estimate_entropy(class_counts, len(Y))
    conditional_entropy = entropy_conditional(x, Y)
    return class_entropy - conditional_entropy

def estimate_entropy(class_counts, N):
    probabilities = class_counts / N
    return -np.sum(probabilities * np.log2(probabilities))

def entropy_conditional(x, y):
    N = len(x)
    B = int(np.floor(np.sqrt(N)))
    y = np.asarray(y)
    x = np.asarray(x)
    x_max, x_min = np.max(x), np.min(x)
    l = (x_max - x_min) / (B - 1)
    
    conditional_entropies = []
    
    for i in range(B):
        bin_values = [y[j] for j, a in enumerate(x) if (i * l) + x_min <= a < (i * l) + l + x_min]
        bin_values = pd.DataFrame(bin_values)
        bin_counts = bin_values.value_counts()
        
        if x_max == x_min:
            bin_counts = []
        
        if len(bin_counts) != 0:
            bin_entropy = estimate_entropy(bin_counts, N)
            bin_prob = sum(bin_counts) / N
            conditional_entropies.append(bin_prob * bin_entropy)
    
    conditional_entropies = np.asarray(conditional_entropies)
    return np.sum(conditional_entropies)

#-----------------------------------------------------------------------
