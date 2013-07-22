__author__ = 'Justin Bayer, Tom Schaul, {justin,tom}@idsia.ch'


import collections
from scipy import array, tile, sum


def crowding_distance(individuals, fitnesses):
    """ Crowding distance-measure for multiple objectives. """
    distances = collections.defaultdict(lambda: 0)
    individuals = list(individuals)
    # Infer the number of objectives by looking at the fitness of the first.
    n_obj = len(fitnesses[individuals[0]])
    for i in xrange(n_obj):
        individuals.sort(key=lambda x: fitnesses[x][i])
        # normalization between 0 and 1.
        normalization = float(fitnesses[individuals[0]][i] - fitnesses[individuals[-1]][i])
        # Make sure the boundary points are always selected.
        distances[individuals[0]] = 1e100
        distances[individuals[-1]] = 1e100
        tripled = zip(individuals, individuals[1:-1], individuals[2:])
        for pre, ind, post in tripled:
            distances[ind] += (fitnesses[pre][i] - fitnesses[post][i]) / normalization
    return distances


def _non_dominated_front_old(iterable, key=lambda x: x, allowequality=True):
    """Return a subset of items from iterable which are not dominated by any
    other item in iterable."""
    items = list(iterable)
    keys = dict((i, key(i)) for i in items)
    dim = len(keys.values()[0])
    if any(dim != len(k) for k in keys.values()):
        raise ValueError("Wrong tuple size.")

    # Make a dictionary that holds the items another item dominates.
    dominations = collections.defaultdict(lambda: [])
    for i in items:
        for j in items:
            if allowequality:
                if all(keys[i][k] < keys[j][k] for k in xrange(dim)):
                    dominations[i].append(j)
            else:
                if all(keys[i][k] <= keys[j][k] for k in xrange(dim)):
                    dominations[i].append(j)

    dominates = lambda i, j: j in dominations[i]

    res = set()
    items = set(items)
    for i in items:
        res.add(i)
        for j in list(res):
            if i is j:
                continue
            if dominates(j, i):
                res.remove(i)
                break
            elif dominates(i, j):
                res.remove(j)
    return res


def _non_dominated_front_fast(iterable, key=lambda x: x, allowequality=True):
    """Return a subset of items from iterable which are not dominated by any
    other item in iterable.

    Faster version.
    """
    items = list(iterable)
    keys = dict((i, key(i)) for i in items)
    dim = len(keys.values()[0])
    dominations = {}
    for i in items:
        for j in items:
            good = True
            if allowequality:
                for k in xrange(dim):
                    if keys[i][k] >= keys[j][k]:
                        good = False
                        break
            else:
                for k in xrange(dim):
                    if keys[i][k] > keys[j][k]:
                        good = False
                        break
            if good:
                dominations[(i, j)] = None
    res = set()
    items = set(items)
    for i in items:
        res.add(i)
        for j in list(res):
            if i is j:
                continue
            if (j, i) in dominations:
                res.remove(i)
                break
            elif (i, j) in dominations:
                res.remove(j)
    return res


def _non_dominated_front_merge(iterable, key=lambda x: x, allowequality=True):
    items = list(iterable)
    l = len(items)
    if l > 20:
        part1 = list(_non_dominated_front_merge(items[:l / 2], key, allowequality))
        part2 = list(_non_dominated_front_merge(items[l / 2:], key, allowequality))
        if len(part1) >= l / 3 or len(part2) >= l / 3:
            return _non_dominated_front_fast(part1 + part2, key, allowequality)
        else:
            return _non_dominated_front_merge(part1 + part2, key, allowequality)
    else:
        return _non_dominated_front_fast(items, key, allowequality)


def _non_dominated_front_arr(iterable, key=lambda x: x, allowequality=True):
    """Return a subset of items from iterable which are not dominated by any
    other item in iterable.

    Faster version, based on boolean matrix manipulations.
    """
    items = list(iterable)
    fits = map(key, items)
    l = len(items)
    x = array(fits)
    a = tile(x, (l, 1, 1))
    b = a.transpose((1, 0, 2))
    if allowequality:
        ndom = sum(a <= b, axis=2)
    else:
        ndom = sum(a < b, axis=2)
    ndom = array(ndom, dtype=bool)
    res = set()
    for ii in range(l):
        res.add(ii)
        for ij in list(res):
            if ii == ij:
                continue
            if not ndom[ij, ii]:
                res.remove(ii)
                break
            elif not ndom[ii, ij]:
                res.remove(ij)
    return set(map(lambda i: items[i], res))


def _non_dominated_front_merge_arr(iterable, key=lambda x: x, allowequality=True):
    items = list(iterable)
    l = len(items)
    if l > 100:
        part1 = list(_non_dominated_front_merge_arr(items[:l / 2], key, allowequality))
        part2 = list(_non_dominated_front_merge_arr(items[l / 2:], key, allowequality))
        if len(part1) >= l / 3 or len(part2) >= l / 3:
            return _non_dominated_front_arr(part1 + part2, key, allowequality)
        else:
            return _non_dominated_front_merge_arr(part1 + part2, key, allowequality)
    else:
        return _non_dominated_front_arr(items, key, allowequality)


non_dominated_front = _non_dominated_front_merge_arr

def non_dominated_sort(iterable, key=lambda x: x, allowequality=True):
    """Return a list that is sorted in a non-dominating fashion.
    Keys have to be n-tuple."""
    items = set(iterable)
    fronts = []
    while items:
        front = non_dominated_front(items, key, allowequality)
        items -= front
        fronts.append(front)
    return fronts
    
''' added by JPQ for Constrained Multi-objective Optimization '''


def _const_non_dominated_front_merge_arr(iterable, key=lambda x: x, allowequality=True):
    items = list(iterable)
    l = len(items)
    if l > 100:
        part1 = list(_const_non_dominated_front_merge_arr(items[:l / 2], key, allowequality))
        part2 = list(_const_non_dominated_front_merge_arr(items[l / 2:], key, allowequality))
        if len(part1) >= l / 3 or len(part2) >= l / 3:
            return _const_non_dominated_front_arr(part1 + part2, key, allowequality)
        else:
            return _const_non_dominated_front_merge_arr(part1 + part2, key, allowequality)
    else:
        return _const_non_dominated_front_arr(items, key, allowequality)
        
def _const_non_dominated_front_arr(iterable, key=lambda x: x, allowequality=True):
    """Return a subset of items from iterable which are not dominated by any
    other item in iterable.

    Faster version, based on boolean matrix manipulations.
    """
    items = list(iterable)  # pop
 
    fits = map(key, items)  # fitness

    x = array([fits[i][0] for i in range(len(fits))])
    v = array([fits[i][1] for i in range(len(fits))])
    c = array([fits[i][2] for i in range(len(fits))])
    
    l = len(items)
    a = tile(x, (l, 1, 1))
    b = a.transpose((1, 0, 2))
    if allowequality:
        ndom = sum(a <= b, axis=2)
    else:
        ndom = sum(a < b, axis=2)
    ndom = array(ndom, dtype=bool)
    res = set()
    for ii in range(l):
        res.add(ii)
        for ij in list(res):
            if ii == ij:
                continue
            if not ndom[ij, ii] and v[ij] and v[ii]:
                res.remove(ii)
                break
            elif not ndom[ii, ij] and v[ij] and v[ii]:
                res.remove(ij)
            elif v[ij] and not v[ii]:
                res.remove(ii)
                break
            elif v[ii] and not v[ij]:
                res.remove(ij)
            elif not v[ii] and not v[ij]:
                cii = abs(sum(c[ii]))
                cij = abs(sum(c[ij]))
                if cii < cij:
                   res.remove(ij)
                else:
                   res.remove(ii)
                   break

    return set(map(lambda i: items[i], res))
    
const_non_dominated_front = _const_non_dominated_front_merge_arr

def const_non_dominated_sort(iterable, key=lambda x: x, allowequality=True):
    """Return a list that is sorted in a non-dominating fashion.
    Keys have to be n-tuple."""
    
    items = set(iterable)
    
    fronts = []
    while items:
        front = const_non_dominated_front(items, key, allowequality)
        items -= front
        fronts.append(front)
    return fronts

def const_crowding_distance(individuals, fitnesses):
    """ Crowding distance-measure for multiple objectives. """
    distances = collections.defaultdict(lambda: 0)
    individuals = list(individuals)
    # Infer the number of objectives by looking at the fitness of the first.
    n_obj = len(fitnesses[individuals[0]][0])
    
    for i in xrange(n_obj):
        individuals.sort(key=lambda x: fitnesses[x][0][i])
        # normalization between 0 and 1.
        normalization = float(fitnesses[individuals[0]][0][i] - fitnesses[individuals[-1]][0][i])
        # Make sure the boundary points are always selected.
        distances[individuals[0]] = 1e100
        distances[individuals[-1]] = 1e100
        tripled = zip(individuals, individuals[1:-1], individuals[2:])
        for pre, ind, post in tripled:
            distances[ind] += (fitnesses[pre][0][i] - fitnesses[post][0][i]) / normalization
    return distances

def const_number_of_feasible_pop(iterable, key=lambda x: x, allowequality=True):
    """Return a subset of items from iterable which are not dominated by any
    other item in iterable.

    Faster version, based on boolean matrix manipulations.
    """
    items = list(iterable)  # pop
 
    fits = map(key, items)  # fitness

    v = list([fits[i][1] for i in range(len(fits))])
    n = v.count(True)
    return n
# ---