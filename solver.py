# Author: Chieh-Chih (George) Yeh
# Date: Oct 10 2022
# Solving the random matrix in various methods
import time
import random
import scipy
import numpy as np
def rand_mat(inp):
    # This function creates a random matrix based on the user-defined properties
    # Read from the input array
    U = inp[0] # Upper limit
    L = inp[1] # Lower limit
    rou = inp[2] # Density of nonzero elements
    n = inp[3] # Number of columns
    m = inp[4] # Number of rows
    z = np.zeros((n, m)) # Create a matrix based on the given dimension
    rang = U - L # The range of the limits
    for i in range(n):
        for j in range(m):
            rand = random.random() # Create a random number between 0 and 1
            if rand > rou: # The possibility of the random number greater than rou is equivalent to the possibility of the element is a zero term
                z[i][j] = 0
            else: # If not then create another random number for the element
                rand2 = random.random()
                scale = rang*rand2 # Since the number only scales between 0 and 1
                mag = L + scale # Equivalent to U - scale
                z[i][j] = mag
    return z

def chk_mat(zz):
    # This function checks if the matrix has one row or column is all 0 
    n = len(zz)
    m = len(zz[0])
    cond = 'yes'
    for i in zz:
        if all([ v == 0. for v in i]):
            cond = 'no'
            break
    for j in range(m):
        col = []
        for k in range(n):
            ele = zz[k][j]
            col.append(ele)
        if all([ v == 0. for v in col]):
            cond = 'no'
    return cond

# Generate the D matrix so the input for Cholesky factor is positive definite
def create_D(zz):
    zt = zz.transpose()
    zza = np.array(zz)
    zta = np.array(zt)
    D = 0.5*(zza + zta)
    return D

# Inverse method required for (3)
def inv_method(zz, b):
    res = np.linalg.inv(zz).dot(b)
    return res

# LU decomposition method for (1)
def LU_method(zz, b):
    lu, piv = scipy.linalg.lu_factor(zz)
    X = scipy.linalg.lu_solve((lu, piv), b)
    return X

# Cholesky factor method for (4)
def Cho_method(zz, b):
    L = scipy.linalg.cholesky(zz, lower=True)
    U = scipy.linalg.cholesky(zz, lower=False)
    Y = np.linalg.inv(L).dot(b)
    X = np.linalg.inv(U).dot(Y)
    return X


def chk_diagonal(zz, b, idm):
    # This function checks if any diagonal elements is zero
    # If so then swap with the following rows if they are nonzero
    # Input the square matrix and the matrix b so the order can be consistent
    n = len(zz)
    m = len(zz[0])
    for i in range(n):
        if zz[i][i] == 0: # if the diagonal is zero
            for j in range(n):# search the rows below to find one that has nonzero
                if zz[j][i] != 0 and zz[i][j] != 0: ######## Fixing point!!!!!!
                    zz[[i, j]] = zz[[j, i]]
                    b[[i, j]] = b[[j, i]]
                    idm[[i, j]] = idm[[j, i]]
                    break
    return zz, b, idm

# This is to generate an identity matrix
def gen_id_mat(n):
    xx = np.zeros((n, n))
    for i in range(n):
        xx[i][i] = 1
    return xx

# Perform Gauss elimination 
# After running this function, the zz matrix will have lower triangle all 0s
def gauss_elim(zz, b, idm):
    for i in range(len(zz)):
        ori_c = zz[i][i]
        for j in range(i+1, len(zz)):
            cc = zz[j][i]/ori_c
            for k in range(i, len(zz)):
                fzz = zz[j][k] - zz[i][k]*cc
                idm[j][k] = idm[j][k] - idm[i][k]*cc
                if abs(fzz) < 1e-08:
                    zz[j][k] = 0
                else:
                    zz[j][k] = fzz
            b[j][0] = b[j][0] - b[i][0]*cc
    return zz, b, idm


################ Main code #######################
# inp = [U, L, density, n, m]
inp = [30, -10, 0.6, 10, 10]
inpb = [50, 0, 0.8, 10, 1]
zz = rand_mat(inp)
ori_z = zz.copy()
n = len(zz)
idm = gen_id_mat(n)
b = rand_mat(inpb)
ori_b = b.copy()
while chk_mat(zz) == 'no':
    zz = rand_mat(inp)

# This part is to make sure the matrix is positive definite
D = create_D(zz)
det = np.linalg.det(D)
kk = np.all(np.linalg.eigvals(D)>0)
kb = 1
while kk == False:
    idd = np.identity(10)
    midd = 30*idd#(k)*idd
    D = D + midd
    kk = np.all(np.linalg.eigvals(D) > 0)
    kb = kb + 1


############# This is to finalize solving the Gauss elimination #############
# This is to get a matrix with nonzero diagonal
func1 = chk_diagonal(zz, b, idm)
zz = func1[0]
b = func1[1]
idm = func1[2]

# This is to get a matrix with all 0 lower triangle
func2 = gauss_elim(zz, b, idm)
zz = func2[0]
b = func2[1]
idm = func2[2]


for i in range(len(zz)):
    ii = len(zz) - i - 1
    cc = zz[ii][ii]
    b[ii][0] = b[ii][0]/cc
    for j in range(len(zz[0])): # This loop is to normalize the whole row so diagonal is 1
        zz[ii][j] = zz[ii][j]/cc
        idm[ii][j] = idm[ii][j]/cc
    for k in range(ii): # This loop is to use the normalized diagonal to subtract other row
        cb = zz[k][ii] 
        zz[k][ii] = zz[k][ii] - cb
        idm[k][ii] = idm[k][ii] - idm[ii][ii]*cb
        b[k][0] = b[k][0] - b[ii][0]*cb
############ This is the end of solving the Gauss elimination #########

zz = [[30.4, 0, -0.675], [0, 30.4, -11.473], [1, -0.1, 0]]
b = [[0], [-30.4], [0]]

########### Presentation of solutions by various methods ################
print(LU_method(zz, b))
print('Solution solved by direct method (via Gauss elimination) for (2)')
print(b)
print('Solution solved by inverse method for (3)')
print(inv_method(zz, b))
