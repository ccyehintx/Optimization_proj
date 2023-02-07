# Author: Chieh-Chih Yeh
# Date: Jan 21 2023
# This is a program to perform dual simplex method
import numpy as np
from numpy.linalg import inv

################# Input #############
c = [1, 1, 0, 0]
A = [[1, 2, -1, 0], [1, 0, 0, -1]]
b = [[2], [1]]
coln = [2, 3] # pick the initial basic variables index
#####################################

def gen_B(A, coln):
    z = len(coln)
    B = np.zeros((z,z))
    for i in range(z):
        for j in range(z):
            B[i][j] = A[i][coln[j]]
    return B

def gen_N(A, c, coln):
    allc = range(len(c))
    nd = len(c) - len(coln)
    cN_idxs = set(allc).difference(coln)
    cN_idx = list(cN_idxs)
    nn = len(cN_idx)
    N = np.zeros((len(A), nn))
    for i in range(len(A)):
        for j in range(nn):
            N[i][j] = A[i][cN_idx[j]]
    return N

def gen_c(c, coln):
    allc = range(len(c))
    cN_idxs = set(allc).difference(coln)
    cN_idx = list(cN_idxs)
    cB_idx = coln
    cB = np.zeros((len(cB_idx), 1))
    cN = np.zeros((len(cN_idx), 1))
    for i in range(len(cB_idx)):
        cB[i][0] = c[cB_idx[i]]
    for j in range(len(cN_idx)):
        cN[j][0] = c[cN_idx[j]]
    return cB, cN

def chk_nonneg(C_N):
    for i in C_N:
        cc = 0
        if i[0] < 0:
            cc = 1
            break
    return cc

def leave_enter(left, coln, main, c_v, cN_idx):
    breakleft = []
    for i in left:
        breakleft.append(i[0])
    minb = min(breakleft)
    minidx = breakleft.index(minb)
    leave_idx = coln[minidx]
    playA = main[minidx] 
    countii = []
    zz = []
    for j in cN_idx:
        if playA[j] < 0:
            countii.append(j)
            zz.append(c_v[j]/abs(playA[j]))
    zmin = min(zz)
    enter_idx = zz.index(zmin)
    return leave_idx, enter_idx

def tableau(c_v, upperl, left, main):
    rr = len(c) + 1
    cnn = len(b) + 1
    tt = np.zeros((cnn, rr))
    tt[0][0] = upperl
    for i in range(len(c_v)):
        ii = i + 1
        tt[0][ii] = c_v[i]
    for i in range(len(left)):
        ii = i + 1
        tt[ii][0] = left[i][0]
    for i in range(len(left)):
        ii = i + 1
        for j in range(len(c_v)):
            jj = j + 1
            tt[ii][jj] = main[i][j]
    return tt

def trans(tt, leave_idx, enter_idx):
    thecol = tt[:, leave_idx+1]
    rowidx = list(range(len(tt)))
    baseidx = 0
    for i in range(len(tt)):
        if thecol[i] != 0:
            den = tt[:, enter_idx+1][i]
            tt[i] = tt[i]/den
            rowidx.remove(i)
            baseidx = i
    for i in rowidx:
        cchange = tt[:, enter_idx+1][i]
        rrr = (cchange - thecol[i])/tt[:, enter_idx+1][baseidx]
        tt[i] = tt[i] - tt[baseidx]*rrr
    return tt

def gen_new(newM):
    left = np.zeros((len(newM[1:, 0]), 1))
    c_v = newM[0, 1:]
    upperl = newM[0][0]
    main = newM[1:, 1:]
    for i in range(len(newM[1:, 0])):
        left[i] = newM[1:, 0][i]
    return left, c_v, upperl, main

# pick the initial basic variables index
cc = 1
while cc == 1: # when the C_N is still not all nonnegative
    B = gen_B(A, coln)
    Bi = inv(B)
    leftn = len(c) - len(coln)
    allc = range(len(c))
    cN_idxs = set(allc).difference(coln)
    cN_idx = list(cN_idxs)
    xB = np.dot(Bi, b)
    xN = np.zeros((leftn, 1))
    cB = gen_c(c, coln)[0]
    cN = gen_c(c, coln)[1]
    N = gen_N(A, c, coln)
    cBt = np.transpose(cB)
    c_v = [] ###
    for i in range(len(c)):
        cbtbv = np.dot(cBt, Bi)
        Am = np.array(A)
        v = c[i] - np.dot(cbtbv, Am[:, i])
        c_v.append(v[0])
    cbtb = np.dot(cBt, Bi)
    upperl = -1*np.dot(cbtb, b)[0][0] ###
    left = np.dot(Bi, b) ###
    main = np.dot(Bi, Am) ###
    cc = chk_nonneg(left) # if == 1 then it's not all nonnegative
    leave_idx = leave_enter(left, coln, main, c_v, cN_idx)[0]
    enter_idx = leave_enter(left, coln, main, c_v, cN_idx)[1]
    tt = tableau(c_v, upperl, left, main) # Current tableau
    if cc == 1:
        incoln = coln.index(leave_idx)
        coln[incoln] = enter_idx
        newM = trans(tt, leave_idx, enter_idx)
        left = gen_new(newM)[0]
        c_v = gen_new(newM)[1]
        upperl = gen_new(newM)[2]
        main = gen_new(newM)[3]


x = np.zeros(len(c))
for i in coln:
    x[i] = xB[coln.index(i)]
print('The optimal solution is=', x)
