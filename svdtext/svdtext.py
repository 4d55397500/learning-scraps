# svdtext.py
"""
   Singular value decomposition and 
   plotting on text 
"""
import numpy as np
import matplotlib.pyplot as plt

def svdtext(filename):
    neighbors = {}
    ix = {}
    i = 0
    with open(filename, 'r') as fl:        
        for ln in fl:            
            words = ln.strip("\n").split()
            for j, w in enumerate(words):
                if not ix.has_key(w):
                    ix[w] = i
                    neighbors[i] = []
                    i += 1
            for j, w in enumerate(words):
                if j > 0:
                    neighbors[ix[w]].append(ix[words[j-1]])
                if j < len(words)-1:
                    neighbors[ix[w]].append(ix[words[j+1]])    
    X = np.zeros((i, i))
    for row in range(i):
        for col in range(i):
            X[row,col] = neighbors[row].count(col)
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    
    fig = plt.figure(figsize=(40,30))
    axes = fig.add_subplot(111)
    for w, q in ix.iteritems():
#        print w
        plt.text(U[q,0], U[q,1], w)
        axes.set_xlim([-0.2, .1])
        axes.set_ylim([-0.1, 0.2])    
    plt.savefig("svd.png")
        
    
if __name__ == "__main__":
    import sys
    svdtext(sys.argv[1])
