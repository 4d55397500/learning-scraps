# minhash.py
import random



def minhash():
    d1 = set(random.randint(0, 2000) for _ in range(1000))
    d2 = set(random.randint(0, 2000) for _ in range(1000))
    jacc_sim = len(d1.intersection(d2)) / len(d1.union(d2))
    print("jaccard similarity: {}".format(jacc_sim))

    N_HASHES = 200
    hash_funcs = []
    for i in range(N_HASHES):
        hash_funcs.append(universal_hashing())

    m1 = [min([h(e) for e in d1]) for h in hash_funcs]
    m2 = [min([h(e) for e in d2]) for h in hash_funcs]
    minhash_sim = sum(int(m1[i] == m2[i]) for i in range(N_HASHES)) / N_HASHES
    print("min-hash similarity: {}".format(minhash_sim))



def universal_hashing():
    def rand_prime():
        while True:
            p = random.randrange(2 ** 32, 2 ** 34, 2)
            if all(p % n != 0 for n in range(3, int((p ** 0.5) + 1), 2)):
                return p
    m = 2 ** 32 - 1
    p = rand_prime()
    a = random.randint(0, p)
    if a % 2 == 0:
        a += 1
    b = random.randint(0, p)
    def h(x):
        return ((a * x + b) % p) % m
    return h




if __name__ == "__main__":
    minhash()


