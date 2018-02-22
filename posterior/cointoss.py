# cointoss.py
"""
 Continuous posterior for
 stream of bernoulli variables

"""
import random


P = 0.7



def uniform_prior():
    return [1.0 / 101 for _ in range(101)]


def posterior(prior):
    c = cointoss()
    theta = [e * 0.01 for e in range(101)]
    pd_theta = theta if c == 1 else [1 - e for e in theta]
    denom = sum([pd_theta[i] * prior[i] for i in range(101)])
    return [pd_theta[i] * prior[i] / denom for i in range(101)]



def cointoss():
    r = random.randint(0, 100)
    if P * 100.0 > r:
        return 1
    return 0


if __name__ == "__main__":
    ps = uniform_prior()
    while True:
        ps = posterior(ps)
        mle, _ = max(enumerate(ps), key=lambda e: e[1])
        print("mle: {} p: {}".format(mle, P))
