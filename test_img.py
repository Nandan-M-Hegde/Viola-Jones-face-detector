import numpy as np
import cv2
import time
from FaceDetect import FaceDetect
import IntegralImage as iimg
import sys
def Scan_subwindow(original_frame, subwindow_dims, fd, features_classifier, zoom_lvl=[0.5, 0.5]):
    #Rescale image
    #frame = original_frame
    frame = cv2.resize(original_frame,None,fx=zoom_lvl[0], fy=zoom_lvl[1], interpolation = cv2.INTER_CUBIC)
    #frame = cv2.resize(original_frame,(19, 19), interpolation = cv2.INTER_AREA)
    #frame = original_frame
    subwindow_width, subwindow_height = subwindow_dims
    frame_width, frame_height = np.shape(frame)
    all_int_imgs = []
    for start_xcoord in range(frame_width - subwindow_width):
        for start_ycoord in range(frame_height - subwindow_height):
            end_xcoord, end_ycoord = start_xcoord + subwindow_width, start_ycoord + subwindow_height
            curr_subwindow = frame[start_xcoord:end_xcoord, start_ycoord:end_ycoord]
            csw = np.copy(curr_subwindow)
            cv2.imshow('Sliding subwindow',cv2.resize(csw, None,fx=10, fy=10, interpolation=cv2.INTER_CUBIC))
            # Display the resulting frame
            curr_int_image = iimg.to_integral_image(curr_subwindow)
            flag = fd.ensemble_votes(curr_int_image, features_classifier)
            orframe = np.copy(original_frame)
            if flag==1:
                curr_subwindow[:,0], curr_subwindow[0,:], curr_subwindow[subwindow_height-1, :], curr_subwindow[:, subwindow_width-1] = 0, 0, 0, 0
                resized_frame = cv2.resize(frame,None,fx=int(1/(zoom_lvl[0]-0.01)), fy=int(1/(zoom_lvl[1]-0.01)), interpolation = cv2.INTER_CUBIC)
                #Display in original frame
                for i in range(np.shape(orframe)[0]):
                    for j in range(np.shape(orframe)[1]):
                        if resized_frame[i][j] == 0:
                            orframe[i][j]=0
                cv2.imshow('Original frame', orframe)
                cv2.waitKey(2000) 
                cv2.imwrite('Images/face_detected.png',orframe)
            #cv2.imshow('frame',cv2.resize(frame,None,fx=int(1/0.08), fy=int(1/0.08), interpolation = cv2.INTER_CUBIC))
            cv2.imshow('Original frame',orframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
               
#cap = cv2.VideoCapture(0)
fd = FaceDetect()
features_classifier = fd.LoadClassifier()
zoom_lvl = [0.09, 0.09]
if len(sys.argv)>1:
    zoom_lvl = [float(sys.argv[1]),float(sys.argv[1])]

fn = 'Images/'+ input("Enter file name: ")
frame = cv2.imread(fn)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
Scan_subwindow(gray, [19,19], fd, features_classifier, zoom_lvl)
cv2.waitKey(0)
cv2.destroyAllWindows()