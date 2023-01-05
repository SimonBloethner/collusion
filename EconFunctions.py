
def demand(p_i, rest, num):
    d = 1 - p_i + (1 / num) * rest * num
    return d


def profit(p_i, d):
    pi = p_i * d
    return pi
