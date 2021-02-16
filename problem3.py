"""
Homework 1 ENPM673: Perception for Autonomous Robots
Problem Number 3
Author: Vivek Sood - 117504279
This python script is used to calculate the svd and 
homography matrix of a given matrix.
"""


import numpy as np


# This function first calculates the single value decomposition matirices and the homography matrix for a given input matrix

# Input:
#   A: input matrix given in the problem: [[-x1, -y1,......., xp1]......[0, 0,.........., yp4]]
# Output:
#   U: First orthogonal matrix formed using eigen vectors of AAt.
#   S: Diagonal matrix.
#   Vt: Transpose of the second orthogonal matrix formed using eigen vectors of AtA.
#   H: Homography matrix of the given input matrix.

def homographyCreator(A):
    At = np.transpose(A)

    AAt = np.matmul(A,At)
    AtA = np.matmul(At,A)

    eValAAtUnsorted, eVecAAt = np.linalg.eig(AAt)
    eValAtA, eVecAtA = np.linalg.eig(AtA)

    eValAAt = eValAAtUnsorted[eValAAtUnsorted.argsort()[::-1]]
    U = eVecAAt[:,eValAAtUnsorted.argsort()[::-1]]
    Vt = np.real(eVecAtA[:,eValAtA.argsort()[::-1]])

    S = np.diag(np.sqrt(eValAAt))
    S = np.concatenate((S,np.zeros((8,1))), axis = 1)

    print("\nEigen Values of AtA \n",eValAtA,"\n")
    #finding the index of the least eigen value in AtA
    leastEigenValue = np.where(np.abs(eValAtA) == np.amin(np.abs(eValAtA)))[0]

    H = Vt[:,leastEigenValue]
    H = np.reshape(Vt[:,leastEigenValue],(3,3))

    return U,S,Vt,H


# main function: first computes the input matrix using the given input paramaters and then 
# calls the svd function using the matrix.

def main():

    x1 =5
    y1 = 5
    xp1 = 100
    yp1 = 100
    x2 = 150
    y2 = 5
    xp2 = 200
    yp2 = 80
    x3 = 150
    y3 = 150
    xp3 = 220
    yp3 = 80
    x4 = 5
    y4 = 150
    xp4 = 100
    yp4 = 200

    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

    A = np.array([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1],
                [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
                [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2],
                [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
                [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3],
                [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
                [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4],
                [0 , 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]])


    print("\nInput Matrix \n",A,"\n")
    U,S,Vt,H = homographyCreator(A)
    print("\nU Matrix \n",U,"\n")
    print("\nS Matrix \n",S,"\n")
    print("\nVt Matrix \n",Vt,"\n")
    print("\nHomography Matrix \n",H,"\n")
if __name__ == '__main__':

    main()