import glob
import numpy as np
import cv2

class LoadImages:
    def __init__(self, fpath, n_i):
        self.pos_is, self.neg_is =[], []
        i=0
        for img in glob.glob(fpath+"face/*.png"):
            if i>n_i:
                break
            i+=1
            self.pos_is.append(cv2.imread(img, 0)) 
        i=0
        for img in glob.glob(fpath+"non_face/*.png"):
            if i>n_i:
                break
            i+=1
            self.neg_is.append(cv2.imread(img, 0))

    def Images(self):
        return [np.array(self.pos_is), np.array(self.neg_is)]