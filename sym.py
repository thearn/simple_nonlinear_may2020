from sympy import *
from itertools import combinations

d, limit = symbols('d limit')

gap = (limit - d)**2

print(diff(gap, d))


nn = 20
nv = 7

ii = 0 
for i in range(nn):
    for k1 in range(nv):
        for k2 in range(k1 + 1, nv):
            ii += 1

print(ii)

nc = nn * nv * (nv - 1)/2.0

print(nc)