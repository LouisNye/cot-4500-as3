import numpy as np
##Question 1
#function given for question 1 euler
def function_to_use(t, y):
    return (t - y**2)
#function to perform euler's method
def eulers_method(t, y, iters, rangevar):
    #initialize step size and perform eulers
    h = (rangevar - t) / iters
    for i in range(iters):
        y = y + (h * function_to_use(t, y))
        t = t + h
    #print inside function
    print("%.5f" % y, "\n")

##Print/call statement for 1
#initialized variables
t = 0
y = 1
iters = 10
rangevar = 2
#calling function for q1, print inside function
eulers_method(t, y, iters, rangevar)

##Question 2
#function given for question 2 range kutta
def function_to_use2(t, y):
    return (t - y**2)
#function to perform range kutta
def runge_kutta(t, y, iters, rangevar):
    #initalize step size and perform range kutta
    h = (rangevar - t) / iters
    for i in range(iters):
        #slope at start
        k_1 = h * function_to_use2(t, y)
        #slope at midpoint
        k_2 = h * function_to_use2((t + (h / 2)), (y + (k_1 / 2)))
        k_3 = h * function_to_use2((t + (h / 2)), (y + (k_2 / 2)))
        #slope at end
        k_4 = h * function_to_use2((t + h), (y + k_3))

        y = y + (1 / 6) * (k_1 + (2 * k_2) + (2 * k_3) + k_4)
        t = t + h
    #print inside function
    print("%.5f" % y, "\n")

##Print/call statement for 2
#initialized variables
t = 0
y = 1
iters = 10
rangevar = 2
#function calling range kutta with initialized vars, print inside function
runge_kutta(t, y, iters, rangevar) 

##Question 3
#function to perform gaussian elimination
def gauss_elim(gauss_mat):
    #acquire matrix length/size
    matlen = gauss_mat.shape[0]
    for i in range(matlen):
        #initialize pivot
        pivot = i
        while gauss_mat[pivot, i] == 0:
            pivot += 1
        gauss_mat[[i, pivot]] = gauss_mat[[pivot, i]]

        #get second row
        for j in range(i + 1, matlen):
            #initialize factor between two rows
            factor = gauss_mat[j, i] / gauss_mat[i, i]
            gauss_mat[j, i:] = gauss_mat[j, i:] - (factor * gauss_mat[i, i:])
    #set up results for print statement
    results = np.zeros(matlen)
    for i in range(matlen - 1, -1, -1):
        results[i] = (gauss_mat[i, -1] - np.dot(gauss_mat[i, i: -1], results[i:])) / gauss_mat[i, i]
    #print inside function
    final_res = np.array([int(results[0]), int(results[1]), int(results[2])], dtype = np.double)
    print(final_res, "\n")

##Print/call statement for 3
#initialize var for func call
gauss_mat = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])
#call func for q3, print inside function
gauss_elim(gauss_mat)

##Question 4 
#function for lu factorization
def lu_factorization(lu_matrix):
    #init matrix length and l u for factorization
    matlen = lu_matrix.shape[0]
    l = np.eye(matlen)
    u = np.zeros_like(lu_matrix)
    #l and u decomposition
    for i in range(matlen):
        for j in range(i, matlen):
            u[i, j] = (lu_matrix[i, j] - np.dot(l[i, :i], u[:i, j]))
        for j in range(i + 1, matlen):
            l[j, i] = (lu_matrix[j, i] - np.dot(l[j, :i], u[:i, i])) / u[i, i]
    #calculating determinant
    determinant = np.linalg.det(lu_matrix)
    #print inside function
    print("%.5f" %determinant, "\n")
    print(l, "\n")
    print(u, "\n")

##Print/call statement for 4
#initialize matrix for lu factorization
lu_matrix = np.array([[1, 1, 0, 3], [2, 1, -1, 1], [3, -1, -1, 2], [-1, 2, 3, -1]], dtype = np.double)
#call lu factorization, print inside function
lu_factorization(lu_matrix)

##Question 5 
#function to find diagonal dominance in matrix
def diag_dom(dnd_matrix, matlen):
    for i in range(0, matlen):
        #initialize total after iterations
        sumiter = 0
        for j in range(0, matlen):
            sumiter = sumiter + abs(dnd_matrix[i][j])
        #find difference betweeen total and abs val of matrix
        sumiter = sumiter - abs(dnd_matrix[i][i])
    #if abs val of matrix is less than sumiter return false else true
    if abs(dnd_matrix[i][i]) < sumiter:
        return False
    else:
        return True

##Print/call statement for 5
#initialize values for function
matlen = 5
dnd_matrix = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3], [4, 2, 3, 12, 2], [3, 2, 4, 0, 8]])
#call function with init vars, print outside function
print(diag_dom(dnd_matrix, matlen))
print()

##Question 6 
def pos_def(pd_matrix):
    #find eigenvalues using np.linalg 
    eigenvals = np.linalg.eigvals(pd_matrix)
    #if all eigenvalues are positive, return true else false
    if np.all(eigenvals > 0):
        return True
    else:
        return False

##Print/call statement for 6
#init matrix to find eigenvals
pd_matrix = np.matrix([[2, 2, 1], [2, 3, 0], [1, 0, 2]])
#call function on matrix, print outside function
print(pos_def(pd_matrix))