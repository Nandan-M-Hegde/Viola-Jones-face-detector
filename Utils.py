from AdaBoost import AdaBoost
from LoadImages import LoadImages
import IntegralImage as iimg
import numpy as np
import pickle

class Utils:

    def CreateClassifier(self, num_features_classifier=20, num_train_imgs=10, train_fpath="Faces_Dataset/train/"):
        #Load images
        limg = LoadImages(train_fpath, num_train_imgs)
        pos_imgs, neg_imgs = limg.Images()
        img_dims = np.shape(pos_imgs[0])
        
        #Form integral images
        pos_iis = [iimg.to_integral_image(img) for img in pos_imgs]
        neg_iis = [iimg.to_integral_image(img) for img in neg_imgs]
        
        #Learn through modified adaboost
        ab = AdaBoost(img_dims)
        
        features_classifier = ab.Learn(pos_iis, neg_iis, num_features_classifier)
        return features_classifier

    def ensemble_votes(self, int_img, features_classifier):
        if sum([feature.get_vote(int_img) for feature in features_classifier]) >= 0:
            return 1
        return -1

    def ensemble_votes_all(self, int_imgs, features_classifier):
        return [self.ensemble_votes(int_img, features_classifier) for int_img in int_imgs]

    def SaveClassifier(self, features_classifier, fn='Features_Classifier.pickle'):
        with open(fn, 'wb') as handle:
            pickle.dump(features_classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def LoadClassifier(self, fn='Features_Classifier.pickle'):
        with open(fn, 'rb') as handle:
            features_classifier = pickle.load(handle)
        return features_classifier

    def EvaluateClassifier(self, features_classifier, num_test_imgs=10, test_fpath="Faces_Dataset/test/"):
        #Load images
        limg = LoadImages(test_fpath, num_test_imgs)
        pos_imgs, neg_imgs = limg.Images()
        img_dims = np.shape(pos_imgs[0])
        
        #Form integral images
        pos_iis = [iimg.to_integral_image(img) for img in pos_imgs]
        neg_iis = [iimg.to_integral_image(img) for img in neg_imgs]
        
        pos_votes = self.ensemble_votes_all(pos_iis, features_classifier)
        neg_votes = self.ensemble_votes_all(neg_iis, features_classifier)
        
        corr=0
        for pvote in pos_votes:
            if pvote == 1:
                corr += 1

        for nvote in neg_votes:
            if nvote == -1:
                corr += 1
        acc = float(corr/float(len(pos_iis)+len(neg_iis)))
        print("Accuracy: {:.2f}%".format(acc*100))

def Driver():
    fd = Utils()
    print("\nTraining Classifier\n")
    num_features_classifier = int(input("Number of features in classifier: "))
    num_train_imgs = int(input("Number of training images: "))
    features_classifier = fd.CreateClassifier(num_features_classifier, num_train_imgs)
    fd.SaveClassifier(features_classifier)
    print("")
