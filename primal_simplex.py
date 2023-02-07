# Author: Chieh-Chih Yeh
# Date: Jan 21 2023
# This is a program to perform primal simplex method
import numpy as np
from numpy.linalg import inv

########### Input ###################
c = [-6, -4, 0, 0, 0]
A = [[1, 1, 1, 0, 0], [2, 1, 0, 1, 0], [2, 3, 0, 0, 1]]
b = [[6], [9], [16]]
coln = [2, 3, 4] # pick the initial basic variables index
###################################

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
        if i < 0:
            cc = 1
            break
    return cc

def gen_dB(C_N, A, Bi, cN_idx):
    minjj = min(C_N)
    jjc = list(C_N).index(minjj)
    jj = cN_idx[jjc]
    Am = np.array(A)
    Aj = Am[:, jj]
    dB = np.dot(-Bi, Aj)
    return dB, jj

def leave_b(dB, xB, coln):
    zz = []
    ctidx = []
    for i in range(len(dB)):
        if dB[i] < 0:
            ctidx.append(coln[i])
            vv = -xB[i]/dB[i]
            zz.append(vv)
    mm = min(zz)
    leaveidx = zz.index(mm)
    return ctidx[leaveidx]

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
    cbtb = np.dot(cBt, Bi)
    C_N = cN - np.transpose(np.dot(cbtb, N))
    cc = chk_nonneg(C_N) # if == 1 then it's not all nonnegative
    if cc == 1:
        dB = gen_dB(C_N, A, Bi, cN_idx)[0]
        enter_idx = gen_dB(C_N, A, Bi, cN_idx)[1]
        leave_idx = leave_b(dB, xB, coln)
        incoln = coln.index(leave_idx)
        coln[incoln] = enter_idx

x = np.zeros(len(c))
for i in coln:
    x[i] = xB[coln.index(i)]
print('The optimal solution is=', x)
