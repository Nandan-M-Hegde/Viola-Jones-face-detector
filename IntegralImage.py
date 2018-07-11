#Integral image calculation for greyscale images
import numpy as np
import cv2

def sum_region(top_left, bottom_right, integral_image):
    #Getting co-ordinates
    tlx, tly = top_left[0], top_left[1]
    brx, bry = bottom_right[0], bottom_right[1]
    #Getting values of the edge points
    A = integral_image[tlx][tly]
    B = integral_image[tlx][bry]
    C = integral_image[brx][tly]
    D = integral_image[brx][bry]
    #Calculating sum
    S = D - B - C + A
    return S

def to_integral_image(img):
    #Getting the shape of img
    m, n = img.shape
    integral_image = np.zeros((m+1, n+1))
    for i in range(1, m+1):
        integral_image[i][1] = img[i-1][0] +  integral_image[i-1][1]
    for j in range(1, n+1):
        integral_image[1][j] = img[0][j-1] +  integral_image[1][j-1] 
    for i in range(2, m+1):
        for j in range(2, n+1):
            integral_image[i][j] = (img[i-1][j-1] +  integral_image[i-1][j] +  integral_image[i][j-1] - integral_image[i-1][j-1])
    return  integral_image