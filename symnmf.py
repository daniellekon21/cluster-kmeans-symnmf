import sys
import math
import numpy as np
import mysymnmf as mk # assuming we have a module named symnmfmodule.c

def_err = "An Error Has Occurred"

# Helper Functions
def initialize_H(N, k, m):
    """
    Initialize the H0 matrix for symNMF.
    H0 is of dimensions N x k, with values uniformly drawn from [0, 2*sqrt(m/k)].
    """
    np.random.seed(1234) 
    H0 = np.random.uniform(low=0, high=2 * math.sqrt(m / k), size=(N, k))
    return H0.tolist()

def print_matrix(A):
    """
    Print a matrix
    Input - A matrix (A)
    """
    for row in A:
        print(','.join(["{:.4f}".format(item) for item in row]))
    return 0

# Main Function
def main():
    args = sys.argv
    if len(args) != 4:
        print(def_err)
        exit(1)
    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]
    except ValueError:
            print(def_err)
            exit(1)
    try:
        data_raw = np.loadtxt(filename, delimiter=',', ndmin=2)
    except Exception:
        print(def_err)
        exit(1)
    if data_raw is None or data_raw.size < 2:
        print(def_err)
        exit(1)
    N = data_raw.shape[0]
    if k >= N:
        print(def_err)
        exit(1)
    data = data_raw.tolist()
    if goal == 'sym':
        print_matrix(sym(data))
    elif goal == 'ddg':
        print_matrix(ddg(data))
    elif goal == 'norm':
        print_matrix(norm(data)) 
    elif goal == 'symnmf':
            if k <= 1:
                print(def_err)
                exit(1)
            W = norm(data) # compute the normalized similarity matrix W
            m = np.mean(W) # Compute m: the average entry of W
            H0 = initialize_H(N, k, m)
            print_matrix(symnmf(k, W, H0))
    else:
            print(def_err)
            exit(1)

def symnmf(k, W, H0):
    return mk.symnmf(W, H0, k)

def norm(data):
    return mk.norm(data)

def ddg(data):
    return mk.ddg(data)

def sym(data):
    return mk.sym(data)

if __name__ == "__main__":
    main()