import numpy as np

import math
def dist(p1, p2):
    d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if d < 1e-10:
        d = 1e-10
    dx = (p1[0] - p2[0]) / d
    dy = (p1[1] - p2[1]) / d
    return d, dx, dy

def solution(x, y):
    a = list(zip(x, y))  
    ax = sorted(a, key=lambda x: x[0])  
    ay = sorted(a, key=lambda x: x[1])  
    
    idx_x = np.argsort(x)
    idx_y = np.argsort(y)

    cache = {}
    p1, p2, mi = min_dist_pair(ax, ay, idx_x, idx_y, cache)  
    
    return mi, p1, p2, cache

def min_dist_pair(ax, ay, idx_x, idx_y, cache={}):
    ln_ax = len(ax)  
    if ln_ax <= 3:
        return brute(ax, idx_x, cache)  

    mid = ln_ax // 2  
    Qx = ax[:mid]  
    Rx = ax[mid:]
    Q_idx_x = idx_x[:mid]
    R_idx_x = idx_x[mid:]

    midpoint = ax[mid][0]  
    Qy = list()
    Ry = list()
    Q_idx_y = list()
    R_idx_y = list()
    for i, x in enumerate(ay):  
        if x[0] <= midpoint:
           Qy.append(x)
           Q_idx_y.append(idx_y[i])
        else:
           Ry.append(x)
           R_idx_y.append(idx_y[i])

    (p1, q1, mi1) = min_dist_pair(Qx, Qy, Q_idx_x, Q_idx_y, cache=cache)
    (p2, q2, mi2) = min_dist_pair(Rx, Ry, R_idx_x, R_idx_y, cache=cache)

    if mi1 <= mi2:
        d = mi1
        mn = (p1, q1)
    else:
        d = mi2
        mn = (p2, q2)

    (p3, q3, mi3) = min_dist_split_pair(ax, ay, idx_y, d, mn, cache=cache)

    if d <= mi3:
        return mn[0], mn[1], d
    else:
        return p3, q3, mi3

def brute(ax, idx_x=[], cache={}):
    if len(idx_x) == 0:
        idx_x = list(range(len(ax)))
    mi, mi_dx, mi_dy = dist(ax[0], ax[1])
    mi_p1 = idx_x[0]
    mi_p2 = idx_x[1]
    ln_ax = len(ax)
    cache[mi_p1, mi_p2] = mi, mi_dx, mi_dy
    
    if ln_ax == 2:
        return mi_p1, mi_p2, mi

    for i in range(ln_ax-1):
        for j in range(i + 1, ln_ax):
            if i != 0 and j != 1:
                d, dx, dy = dist(ax[i], ax[j])
                p1, p2 = idx_x[i], idx_x[j]
                cache[p1, p2] = d, dx, dy
                if d < mi:  
                    mi = d
                    mi_p1, mi_p2 = idx_x[i], idx_x[j]
    return mi_p1, mi_p2, mi


def min_dist_split_pair(p_x, p_y, idx_y, delta, best_pair, cache={}):
    ln_x = len(p_x)  
    mx_x = p_x[ln_x // 2][0]  
    
    s_y = []
    idx_yy = []
    for i, x in enumerate(p_y):
        if mx_x - delta <= x[0] <= mx_x + delta:
            s_y.append(x)
            idx_yy.append(idx_y[i])

    # print("!!!")
    # print(idx_yy)
    # print("???")
    best = delta  
    
    ln_y = len(s_y) 
    
    for i in range(ln_y - 1):
        for j in range(i+1, min(i + 7, ln_y)):
            p, q = s_y[i], s_y[j]
            dst, dst_x, dst_y = dist(p, q)
            i1, i2 = idx_yy[i], idx_yy[j]
            cache[i1, i2] = dst, dst_x, dst_y
            if dst < best:
                best_pair = i1, i2
                best = dst
    return best_pair[0], best_pair[1], best


if __name__ == '__main__':
    

    import time

    n = 500
    np.random.seed(2)
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)

    t = time.time()
    mn, i, k, cache = solution(x, y)
    print("sol:", i, k, mn)
    print("time:", time.time() - t)
    print()
    for i, k in cache:
        print("test:", cache[i, k][0], np.sqrt((x[i] - x[k])**2 + (y[i] - y[k])**2))
    quit()
    print(len(cache))
    # for a,b in cache:
    #     if (b, a) in cache:
    #         print("duplicate!")
    #     #print(a, b, cache[a,b])
    print()

    t = time.time()
    print(brute(list(zip(x, y))))
    print(time.time() - t)
