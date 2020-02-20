def correlation(x,y):
    fracA = sum((x-np.average(x))*(y-np.average(y)))
    fracB = np.sqrt(sum(np.power(x-np.average(x),2)))*np.sqrt(sum(np.power(y-np.average(y),2)))
    return fracA/fracB

def minkowski_distance(x,y,p):
    return np.sum(np.abs(x-y)**p)**(1/p)

def euclidianne_distance(x,y):
    return minkowski_distance(x,y,2)

def manhathan_distance(x,y):
    return minkowski_distance(x,y,1)

def lim_plus_distance(x,y):
    return np.max(np.abs(x-y))

def lim_min_distance(x,y):
    return np.min(np.abs(x-y))

def cosine(x,y):
    return np.sum(x*y)/(np.sqrt(np.sum(x**2))*np.sqrt(np.sum(y**2)))