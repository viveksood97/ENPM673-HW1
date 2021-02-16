# ENPM673-HW1
Homework 1 for  ENPM673: Perception for Autonomous Robots

## Problem 1
Problem 1's solution is inside homeworkReport.pdf

## Problem 2
#### Description
The script is used to collect points from input video and use the points to generate the curve that best fits those points using Least Square, Total Least Square, RANSAC.
#### Packages used:
```
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import argparse
```
#### To run the code use the following command:
```bash
python3 problem2.py --pathToVideo /{directory with the input video}/{filename}.mp4
```
Default pathToVideo is ./video/Ball_travel_2_updated.mp4
#### Note!!!
Please close the plot window after observing or saving the plot to present the next plot.
## Problem 3
The script generates homography matrix for the given input. The mathematical solution is inside the report
#### Packages used:
```
import numpy as np
```
#### To run the code use the following command:
```bash
python3 problem3.py 
