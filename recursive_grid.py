import numpy as np
import time

def min_dist_brute(x, y, limit=0.05):
    n = x.size
    
    md = 1e9
    pts = (0, 0)

    pts_bad = {}

    for i in range(n):
        for k in range(i + 1, n):
            d = np.sqrt((x[i] - x[k])**2 + (y[i] - y[k])**2)
            if d < md:
                md = d
                pts = [i, k]

            #if d < limit:
            if (i, k) not in pts_bad and (k, i) not in pts_bad:
                dx = (x[i] - x[k]) / d
                dy = (y[i] - y[k]) / d
                pts_bad[i, k] = [d, dx, dy]

    return md, pts, pts_bad

def min_dist_grid(x, y, top=True, limit=0.05):
    n = x.size
    if n <= 3:
        return min_dist_brute(x, y, limit=limit)

    idx_sort = range(n)
    if top:
        idx_sort = np.argsort(x)

    i_div = x.size//2

    idx_sort_l = idx_sort[:i_div]
    idx_sort_r = idx_sort[i_div:]

    xl = x[idx_sort_l]
    xr = x[idx_sort_r]

    L = (xl[-1] + xr[0])/2.0

    yl = y[idx_sort_l]
    yr = y[idx_sort_r]

    md_l, pts_l, pts_bad_l = min_dist_grid(xl, yl, top=False, limit=limit)
    md_r, pts_r, pts_bad_r = min_dist_grid(xr, yr, top=False, limit=limit)

    pts_bad = {}
    for a, b in pts_bad_l:
        ii, kk = idx_sort_l[a], idx_sort_l[b]
        if (ii, kk) not in pts_bad and (kk, ii) not in pts_bad:
            pts_bad[ii, kk] = pts_bad_l[a, b]

    for a, b in pts_bad_r:
        ii, kk = idx_sort_r[a], idx_sort_r[b]
        if (ii, kk) not in pts_bad and (kk, ii) not in pts_bad:
            pts_bad[ii, kk] = pts_bad_r[a, b]

    epsilon = md_l
    pts = pts_l
    idx_lookup = idx_sort_l
    if md_r < md_l:
        epsilon = md_r
        pts = pts_r
        idx_lookup = idx_sort_r
    
    pts = (idx_lookup[pts[0]], idx_lookup[pts[1]])
    
    # closest pair in grid epsilon
    idx_strip = np.where(abs(L - x) <= epsilon)

    x_strip = x[idx_strip]
    y_strip = y[idx_strip]

    nn = x_strip.size
    min_d = epsilon
    for i in range(nn):
        for k in range(i + 1, min(i + 7, nn)):
            d = np.sqrt((x_strip[i] - x_strip[k])**2 + (y_strip[i] - y_strip[k])**2)
            if d < min_d:
                min_d = d
                idx1, idx2 = idx_strip[0][i], idx_strip[0][k]
                pts = (idx1, idx2)

            #if d < limit:
            idx1, idx2 = idx_strip[0][i], idx_strip[0][k]
            if (idx1, idx2) not in pts_bad and (idx2, idx1) not in pts_bad:
                dx = (x_strip[i] - x_strip[k]) / d
                dy = (y_strip[i] - y_strip[k]) / d
                pts_bad[idx1, idx2] = [d, dx, dy]


    return min_d, pts, pts_bad

if __name__ == '__main__':
    print()
    np.random.seed(0)
    n = 5000
    limit=10.0
    x = np.random.uniform(0, 100, n)
    y = np.random.uniform(0, 100, n)


    t = time.time()
    m, p, pts_bad = min_dist_grid(x, y, top=True, limit=limit)
    #for a, b in pts_bad:
    #    print(a, b, pts_bad[a, b])
    
    print("recursive:", m, p)
    print("pts_bad", len(pts_bad))
    print("time", time.time() - t)
    print()

    t = time.time()
    m, p, pts_bad = min_dist_brute(x, y, limit=limit)
    #for a, b in pts_bad:
    #    print(a, b, pts_bad[a, b])

    print("brute:", m, p)
    print("pts_bad", len(pts_bad))
    print("time", time.time() - t)


