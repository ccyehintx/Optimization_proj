# Author: Chieh-Chih (George) Yeh
# 2022/10/22
# This is for converting the original matrix into compressed form
# Then perform the multiplication without unzipping them
import random
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

def compressed_mat(zz):
    v1 = [] # Creating the space for the first vector
    v2 = [] # Creating the space for the second vector
    v3 = [] # Creating the space for the third vector
    no_col = len(zz[0]) # Number of columns
    for i in range(no_col):
        count = 0 # This is to count the nonzero elements
        non0idx = [] # This is to note down the indices of the rows with nonzero terms
        valset = [] # This is to note down the actual value of the nonzero term sorted by rows
        for j in range(len(zz)):
            val = zz[j][i]
            if val != 0:
                count = count + 1
                non0idx.append(j)
                valset.append(val)
        v1.append(count)
        v2.append(non0idx)
        v3.append(valset)
    return [v1, v2, v3]

def mul_comp(cv1, cv2):
    # This function is to perform multiplication of the matrices in compressed forms
    v2_1 = cv1[1]
    # As the number of rows cannot be obtained directly from the compressed form of matrix
    # But it's required that all columns and rows cannot be all zero
    # So the largest number in all sets in v2_1 will give the row number - 1
    hh = 0
    for h in v2_1:
        if max(h) > hh:
            hh = max(h)
    n_row = hh + 1 # Number of rows for the multiplied matrix
    v3_1 = cv1[2]
    v1_2 = cv2[0]
    n_col = len(v1_2) # number of columns for the multiplied matrix
    v2_2 = cv2[1]
    v3_2 = cv2[2]
    size = n_row*n_col # Size of the multiplied matrix
    zz = np.zeros((n_row, n_col)) # Creating a zero matrix for the multiplied matrix
    for i in range(size):
        ii = i + 1
        mag = 0
        rowth = i//n_col # call the row from the first matrix
        colth = ii%n_col - 1 # call the column from the second matix
        # then first check the second matrix's column
        non0m2 = v2_2[colth] # here store the indices of nonzero terms, next check if M1(row) has
        for j in non0m2:
            jth = non0m2.index(j) # The position of the number in v2 so can extract number from v3
            for k in range(len(v2_1[j])):
                if v2_1[j][k] == rowth: # If the required index equal to the idx of row
                    mag = mag + v3_1[j][k]*v3_2[colth][jth] # The multiplication process
        zz[rowth][colth] = mag # Fill into the number

    return zz


################ Create the 2 random matices #######################
# inp = [U, L, density, n, m]
inp1 = [20, -10, 0.4, 10, 16]
inp2 = [20, -10, 0.4, 16, 9]
zz1 = rand_mat(inp1)
zz2 = rand_mat(inp2)
# This while loop will only stop when there is no all 0 element in one column or row
while chk_mat(zz1) == 'no':
    zz1 = rand_mat(inp1)
while chk_mat(zz2) == 'no':
    zz2 = rand_mat(inp2)
############### Full original matrices have been created #############
cc1 = compressed_mat(zz1) # This produces the compressed form of first matrix
cc2 = compressed_mat(zz2) # This produces the compressed form of second matrix
ccm = mul_comp(cc1, cc2) # This multiplies the compressed form of the two vectors
print('Compressed form of the first matrix')
print(cc1)
print('Compressed form of the second matrix')
print(cc2)
print('Multiply the two compressed form')
print(ccm)
print('Multiply the two original matrix (for comparison)')
print(np.matmul(zz1, zz2))
