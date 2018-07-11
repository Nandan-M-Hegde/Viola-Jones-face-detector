import numpy as np
from HaarFeatures import HaarFeatures
import IntegralImage as iimg
from GenerateFeatures import GenerateFeatures
from LoadImages import LoadImages

class AdaBoost:
    def __init__(self, img_dims, min_feature_dimensions=[-1,-1], max_feature_dimensions=[-1,-1]):
        #Create all features
        gf = GenerateFeatures(img_dims, min_feature_dimensions, max_feature_dimensions)
        self.all_features = gf.Features()
        self.num_features = len(self.all_features)
        self.feature_ids = list(range(self.num_features))
        
    def Initialize_weights(self, num_positives, num_negatives):
        pos_weights = np.ones((num_positives))/float(2*num_positives)
        neg_weights = np.ones((num_negatives))/float(2*num_negatives)
        self.weights_imgs = np.hstack((pos_weights, neg_weights))

    def Normalize_weights(self):
        total_weights = sum(self.weights_imgs)
        for i in range(len(self.weights_imgs)):
            self.weights_imgs[i] = float(self.weights_imgs[i]/total_weights)

    def Learn(self, pos_iis, neg_iis, num_features_classifier=-1):
        #Feature dimensions -> [width, height]    
        
        #Parameters initialization
        num_positives, num_negatives = len(pos_iis), len(neg_iis)
        num_images = num_positives + num_negatives
        img_height, img_width = np.shape(pos_iis[0])
        
        #Concatenate all labels and images
        labels = np.array([1]*num_positives + [-1]*num_negatives)
        images = pos_iis + neg_iis

        #Initialize weights
        self.Initialize_weights(num_positives, num_negatives)
        
        if num_features_classifier==-1:
            num_features_classifier = self.num_features
        if num_features_classifier < 1 and num_features_classifier>0:
            num_features_classifier = int(num_features_classifier*float(self.num_features))
        
        #Votes for all features across all images
        all_votes = []

        #Compute all votes
        for i in range(num_images):
            votes_img = []
            for feature in self.all_features:
                votes_img.append(feature.get_vote(images[i]))
            all_votes.append(votes_img)
        votes = np.array(all_votes)
        votes_all_imgs = np.transpose(votes)
        '''
        print("No of features: {}".format(self.num_features))
        print("Number of features_classifier: {}".format(num_features_classifier))
        print("Images shape: {}".format(np.shape(images)))
        print("Votes shape: {}".format(np.shape(votes)))
        '''
        #Selecting features_classifier
        features_classifier = []
        for Q in range(num_features_classifier):
            #Intialize
            classification_errors = []
            #Normalize weights
            self.Normalize_weights()
            j=0
            for feature in self.all_features:
                error = 0
                for img_id in range(num_images):
                    if labels[img_id] != votes_all_imgs[j][img_id]:
                        error += self.weights_imgs[img_id]
                classification_errors.append(error)
                j+=1

            #Minimum error feature
            best_feature_id = self.feature_ids[np.argmin(classification_errors)]
            best_error = classification_errors[np.argmin(classification_errors)]
            beta = float(best_error/(1-best_error))

            #Best feature
            best_feature = self.all_features[best_feature_id]

            #Update weights
            for img_id in range(num_images):
                if labels[img_id] == votes[img_id][best_feature_id]:
                    self.weights_imgs[img_id] *= beta

            print("Iteration: {}\tBest feature: {}".format(Q, best_feature.feature_position))
            #Remove current feature
            self.feature_ids.remove(best_feature_id)
            self.all_features.remove(best_feature)
            features_classifier.append(best_feature)
        
        self.num_features -= num_features_classifier
        return features_classifier

def Driver():
    #Load images
    limg = LoadImages("Faces_Dataset/train/", 10)
    pos_imgs, neg_imgs = limg.Images()
    img_dims = np.shape(pos_imgs[0])
    
    #Form integral images
    pos_iis = [iimg.to_integral_image(img) for img in pos_imgs]
    neg_iis = [iimg.to_integral_image(img) for img in neg_imgs]
    
    #Learn through modified adaboost
    ab = AdaBoost(img_dims)
    for num_features_classifier in [2, 5, 10]:
        ab.Learn(pos_iis, neg_iis, num_features_classifier)

#Driver()