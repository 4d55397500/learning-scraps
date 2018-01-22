# minhash.py
import random
import time

def minhash():

    DOC_SIZE = 10**4
    N_DOCS = 100

    print("Generating mock documents...")
    docs = [[random.randint(0, 10**6) for _ in range(DOC_SIZE)] for _ \
            in range(N_DOCS)]
    print("Generating shingles and shingle hashes...")
    shingles = lambda d: [hash(tuple(d[i:i+j])) for j in range(3) for i in range(len(d)-j)]
    doc_hashes = [set(shingles(d)) for d in docs]
    print("Computing Jaccard similarities..")
    jacc_sims = {}

    n, n_executions = 0, 20
    total_time = 0.0
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            if n > n_executions:
                break
            start = time.time()
            s1, s2 = doc_hashes[i], doc_hashes[j]
            jacc_sims[(i,j)] = len(s1.intersection(s2)) * 1.0 / len(s1.union(s2))
            end = time.time()
            total_time += end - start
            n += 1
            n_executions += 1
    print("{} seconds elapsed on average for computing the Jaccard similarity directly".format(
       total_time * 1.0 / n))






if __name__ == "__main__":
    minhash()


