"""
Homework 1 ENPM673: Perception for Autonomous Robots
Problem Number 2
Author: Vivek Sood - 117504279
This python script is used to input object position
from an input video and curve fit this data using 
Least Square Method(LS), Total Least Square Method
(TLS) and Random Sample Consensus(RANSAC).
"""


#Importing all the required packages and libraries

import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import argparse


# Transforms the input data points list to matrices(numpy arrays) where (x, y) is transformed to ((x**2, x, 1), y)

# Input:
#   inputCorr: list with each index containing a coordinate pair tuple: [(x1, y1), (x2, y2),........, (xn, yn)]
# Output:
#   ax: numpy array containing the transformed x-coordinate : [[x1**2, x1, 1], [x2**2, x2, 1],........, [xn**2, xn, 1]]
#   ax: numpy array containing the y-coordinate : [[y1], [y2],........, [yn]]

def matrixTransformer(inputCorr):
    shape = len(inputCorr)

    ax = np.zeros(shape=(shape, 3))
    ay = np.zeros(shape=(shape, 1))

    for i, point in enumerate(inputCorr):
        ax[i] = [np.power(point[0], 2), point[0], 1]
        ay[i] = [point[1]]
    return ax, ay

def modelForCurve(inputCorr,coefficientMatrix):
    dataSetLength = len(inputCorr)

    ay = np.zeros(shape=(dataSetLength, 1))

    for i, point in enumerate(inputCorr):
        ay[i] = coefficientMatrix[0]*(np.power(point[0],2)) + coefficientMatrix[1]*point[0] + coefficientMatrix[2]
    return ay


# Converts the given input points into points that fit a curve using total least square curve fitting algorithm

# Input:
#   inputCorr: list with each index containing a coordinate pair tuple: [(x1, y1), (x2, y2),........, (xn, yn)]
# Output:
#   transformedY: numpy array containing the transformed value of y-coordinate after substituting value of 
#                 y-coordinate in the model : [[a*x1**2 + bx1 + c], [a*x2**2 + bx2 + c],........, [a*xn**2 + bxn + c]]

def totalLeastSquare(inputCorr):
    ax, ay = matrixTransformer(inputCorr)

    Z = np.concatenate((ax, ay), axis=1)

    U, S, V = np.linalg.svd(Z)
    V = np.transpose(V)
    n = ax.shape[1]

    VXY = V[0:n, n:] 
    VYY = V[n: , n: ]

    B = np.dot(-VXY, np.linalg.pinv(VYY))
    coefficientMatrix = B.transpose()[0]

    error = meanError(inputCorr,coefficientMatrix)
    print("Total Least Square Coefficients ----->", coefficientMatrix, "with error =",error,"\n")
    transformedY = ax.dot(coefficientMatrix)
    return transformedY


# Converts the given input points into points that fit a curve using least square curve fitting algorithm

# Input:
#   inputCorr: list with each index containing a coordinate pair tuple: [(x1, y1), (x2, y2),........, (xn, yn)]
# Output:
#   transformedY: numpy array containing the transformed value of y-coordinate after substituting value of 
#                 y-coordinate in the model : [[a*x1**2 + bx1 + c], [a*x2**2 + bx2 + c],........, [a*xn**2 + bxn + c]]

def leastSquare(inputCorr):
    ax, ay = matrixTransformer(inputCorr)
    axT = ax.transpose()

    first = axT.dot(ax)
    firstInv = np.linalg.pinv(first)

    second = axT.dot(ay)
    finalBeforeTranspose = firstInv.dot(second)

    coefficientMatrix = finalBeforeTranspose.transpose()[0]
    error = meanError(inputCorr,coefficientMatrix)
    print("Least Square Coefficients ----->", coefficientMatrix, "with error =",error,"\n")
    transformedY = ax.dot(coefficientMatrix)
    return transformedY


# Converts the given input points into points that fit a curve using least square curve fitting algorithm

# Input:
#   chosen_points: list of three point chosen at random from the total input points : [(x1,y1), (x2,y2), (x3,y3)]
# Output:
#   coefficientMatrix: matrix containing the values of (a,b,c) in the equation a*x**2 + b*x + c for the given input points: [a, b, c]

def findCoeficeients(chosen_points):
    ax, ay = matrixTransformer(chosen_points)

    aInv = np.linalg.pinv(ax)

    coefficientMatrix = np.transpose(np.dot(aInv,ay))[0]
    return coefficientMatrix


# Converts the given input points into points that fit a curve using least square curve fitting algorithm

# Input:
#   input_points_filtered: list of points from the input dataset apart from the three points chosen at random: [(x1, y1), (x2, y2),........, (xn, yn)]
#   coefficientMatrix:  matrix containing the values of (a,b,c) in the equation a*x**2 + b*x + c for the three points chosen at random: [a, b, c]
#   threshold: the error threshold that specifies whether a point is an inlier or outlier : 55
#   minInliers: Minimum number of inliers that is essential for an acceptable for a curve fit: 10
# Output:
#   inliers: list of points that qualify as inliers: [(x1, y1), (x2, y2),........, (xi, yi)]

def findError(input_points_filtered,coefficientMatrix,threshold,minInliers):
    inliers = []

    for point in input_points_filtered:
        y = coefficientMatrix[0]*(np.power(point[0],2)) + coefficientMatrix[1]*point[0] + coefficientMatrix[2]
        error = abs(y-point[1])

        if(error < threshold):
            inliers.append(point)
    return inliers


#Converts the given input points into points that fit a curve using least square curve fitting algorithm

# Input:
#   input_points: list with each index containing a coordinate pair tuple: [(x1, y1), (x2, y2),........, (xn, yn)]
#   coefficientMatrix:  matrix containing the values of (a,b,c) in the equation a*x**2 + b*x + c for the three points chosen at random: [a, b, c]
# Output:
#   e: mean error of all the points.: float

def meanError(input_points,coefficientMatrix):
    error = 0
    for point in input_points:
        y = coefficientMatrix[0]*(np.power(point[0],2)) + coefficientMatrix[1]*point[0] + coefficientMatrix[2]
        error += abs(y-point[1])

    e = error/len(input_points) 
    return e
    

# Converts the given input points into points that fit a curve using RANSAC curve fitting algorithm

# Input:
#   inputCorr: list with each index containing a coordinate pair tuple: [(x1, y1), (x2, y2),........, (xn, yn)]
#   iterations: Number of iterations that the function will run for : 1000
#   threshold: the error threshold that specifies whether a point is an inlier or outlier : 150
#   minInliers: Minimum number of inliers that is essential for an acceptable for a curve fit: 70% of the length of input
# Output:
#   transformedY: numpy array containing the transformed value of y-coordinate after substituting value of 
#                 y-coordinate in the model : [[a*x1**2 + bx1 + c], [a*x2**2 + bx2 + c],........, [a*xn**2 + bxn + c]]
#   flag: boolean value representing whether RANSAC algorithm gave a successful output that satisfy the given input parameters: True or False

def ransac(inputCorr, iterations, threshold, minInliers):
    flag = True
    error = float('inf')
    datasetLength = len(inputCorr)

    ay = np.zeros(shape=(datasetLength, 1))

    for _ in range(iterations):
        chosen_points = random.sample(inputCorr, 5)

        input_points_filtered = inputCorr[:]

        # remove the points we are using from this iteration of the loop
        input_points_filtered.remove(chosen_points[0])
        input_points_filtered.remove(chosen_points[1])
        input_points_filtered.remove(chosen_points[2])
        input_points_filtered.remove(chosen_points[3])
        input_points_filtered.remove(chosen_points[4])

        coefficientMatrix = findCoeficeients(chosen_points)
        inliers = findError(input_points_filtered,coefficientMatrix,threshold,minInliers)
        currentError = meanError(inputCorr,coefficientMatrix)
        if(len(inliers) > minInliers):
            if(currentError < error):
                error = currentError
                ax, ay= matrixTransformer(inputCorr)
                ay = ax.dot(coefficientMatrix)

    if(np.count_nonzero(ay) != datasetLength):
        flag = False

    print("RANSAC Coefficients ----->", coefficientMatrix ,"with error =",error,"\n")
    return ay, flag


# Input:
#   x: input x co-ordinate: [x1, x2, x3,......, xn]
#   y: input y co-ordinate: [y1, y2, y3,......, yn]
#   finalY: values of y for given x values and determined values of equation coeficients: [yf1, yf2, yf3,......, yfn]
#   label: label for the curve ie the type of algorithm used to plot the curve: RANSAC or LS or TLS
#   title: title of curve

def plotPlots(x,y,finalY,label,title):
    plt.scatter(x, y, label='data',marker='x',color='black')
    plt.plot(x, finalY, label=label,color='black',linewidth=2)
    plt.title(title)
    plt.legend(loc="upper left")
    plt.show()


# main function: captures the points from the input video    

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--pathToVideo', default='./video/Ball_travel_2_updated.mp4', help=', Default: ./video/Ball_travel_2_updated.mp4')
    Args = Parser.parse_args()
    video = Args.pathToVideo
    print("\n")
    # converts format of numbers from exponential to decimal 
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})


    #add path for both input videos as a raw string
    #inputVideos = [r"Ball_travel_10fps.mp4",r"Ball_travel_2_updated.mp4"]

    


    cap = cv2.VideoCapture(video)
    inputCorr = []
    try:
        while(cap.isOpened()):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                # compute the center of the contour
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            inputCorr.append((cX, thresh.shape[1]-cY))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        xCorr, yCorr = zip(*inputCorr)
        print("Output for",video.split("/")[-1],"\n")
        print("#"*50,"\n")
        lsY = leastSquare(inputCorr)
        plotPlots(xCorr,yCorr,lsY,"LS","LS For "+video.split("/")[-1])

        tlsY = totalLeastSquare(inputCorr)
        plotPlots(xCorr,yCorr,tlsY,"TLS","TLS For "+video.split("/")[-1])
        
        # minimum number of inliers is to 70% as the dataset is very small
        minmumInliners = int(0.70 * len(inputCorr))
        RANSACy, flag = ransac(inputCorr,1000,150,minmumInliners)
        if(flag):
            plotPlots(xCorr,yCorr,RANSACy,"RANSAC","RANSAC For "+video.split("/")[-1])
        else:
            print("RANSAC cant fit a curve for the given data and parameters","\n")
        print("#"*50,"\n")
        
        
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    main()




