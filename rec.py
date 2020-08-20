import numpy as np

import math
def dist(p1, p2):
    d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    dx = (p1[0] - p2[0]) / d
    dy = (p1[1] - p2[1]) / d
    return d, dx, dy

def solution(x, y):
    a = list(zip(x, y))  # This produces list of tuples
    ax = sorted(a, key=lambda x: x[0])  # Presorting x-wise
    ay = sorted(a, key=lambda x: x[1])  # Presorting y-wise
    
    idx_x = np.argsort(x)
    idx_y = np.argsort(y)

    p1, p2, mi = closest_pair(ax, ay, idx_x, idx_y)  # Recursive D&C function
    
    return mi, p1, p2

def closest_pair(ax, ay, idx_x, idx_y):
    ln_ax = len(ax)  # It's quicker to assign variable
    if ln_ax <= 3:
        return brute(ax, idx_x)  # A call to bruteforce comparison

    mid = ln_ax // 2  # Division without remainder, need int
    Qx = ax[:mid]  # Two-part split
    Rx = ax[mid:]
    Q_idx_x = idx_x[:mid]
    R_idx_x = idx_x[mid:]

    # Determine midpoint on x-axis
    midpoint = ax[mid][0]  
    Qy = list()
    Ry = list()
    Q_idx_y = list()
    R_idx_y = list()
    for i, x in enumerate(ay):  # split ay into 2 arrays using midpoint
        if x[0] <= midpoint:
           Qy.append(x)
           Q_idx_y.append(idx_y[i])
        else:
           Ry.append(x)
           R_idx_y.append(idx_y[i])

    # Call recursively both arrays after split
    (p1, q1, mi1) = closest_pair(Qx, Qy, Q_idx_x, Q_idx_y)
    (p2, q2, mi2) = closest_pair(Rx, Ry, R_idx_x, R_idx_y)

    # Determine smaller distance between points of 2 arrays
    if mi1 <= mi2:
        d = mi1
        mn = (p1, q1)
    else:
        d = mi2
        mn = (p2, q2)

    # Call function to account for points on the boundary
    (p3, q3, mi3) = closest_split_pair(ax, ay, idx_y, d, mn)

    # Determine smallest distance for the array
    if d <= mi3:
        return mn[0], mn[1], d
    else:
        return p3, q3, mi3

def brute(ax, idx_x=[]):
    if len(idx_x) == 0:
        idx_x = list(range(len(ax)))
    mi, mi_dx, mi_dy = dist(ax[0], ax[1])
    p1 = idx_x[0]
    p2 = idx_x[1]
    ln_ax = len(ax)

    if ln_ax == 2:
        return p1, p2, mi

    for i in range(ln_ax-1):
        for j in range(i + 1, ln_ax):
            if i != 0 and j != 1:
                d, dx, dy = dist(ax[i], ax[j])
                if d < mi:  # Update min_dist and points
                    mi = d
                    p1, p2 = idx_x[i], idx_x[j]
    return p1, p2, mi


def closest_split_pair(p_x, p_y, idx_y, delta, best_pair):
    ln_x = len(p_x)  # store length - quicker
    mx_x = p_x[ln_x // 2][0]  # select midpoint on x-sorted array

    # Create a subarray of points not further than delta from
    # midpoint on x-sorted array
    
    #s_y = [x for x in p_y if mx_x - delta <= x[0] <= mx_x + delta]
    
    s_y = []
    idx_yy = []
    for i, x in enumerate(p_y):
        if mx_x - delta <= x[0] <= mx_x + delta:
            s_y.append(x)
            idx_yy.append(idx_y[i])

    # print("!!!")
    # print(idx_yy)
    # print("???")
    best = delta  # assign best value to delta
    
    ln_y = len(s_y)  # store length of subarray for quickness
    
    for i in range(ln_y - 1):
        for j in range(i+1, min(i + 7, ln_y)):
            p, q = s_y[i], s_y[j]
            dst, dst_x, dst_y = dist(p, q)
            if dst < best:
                best_pair = idx_yy[i], idx_yy[j] 
                best = dst
    return best_pair[0], best_pair[1], best


import time

n = 50
np.random.seed(40)
x = np.random.uniform(0, 100, n)
y = np.random.uniform(0, 100, n)

t = time.time()
mn, i, k = solution(x, y)

print("sol:", i, k, mn)
print("time:", time.time() - t)
print()
print("test:", np.sqrt((x[i] - x[k])**2 + (y[i] - y[k])**2))


print()

t = time.time()
print(brute(list(zip(x, y))))
print(time.time() - t)
