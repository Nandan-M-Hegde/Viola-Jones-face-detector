from HaarFeatures import HaarFeatures

class GenerateFeatures:

    def __init__(self, img_dimensions, min_feature_dimensions=[-1,-1], max_feature_dimensions=[-1,-1]):
        #dimensions -> [width, height]
        img_width, img_height = img_dimensions[0], img_dimensions[1]

        #Default feature dimensions for different feature types
        minfdims = [[2,1],[1,2],[3,1],[1,3],[2,2]]
        maxfdims = [[img_dimensions[0] ,img_dimensions[1]] for i in range(5)]
        fgrowth = minfdims

        #Set to defaults
        if(min_feature_dimensions==[-1,-1] and max_feature_dimensions==[-1,-1]):
            min_feature_dimensions = minfdims                                        
            max_feature_dimensions = maxfdims                                        
        
        feature_growth = min_feature_dimensions
        self.all_features = []

        for feature_type in range(5):       #all feature types
            for feature_width in range(min_feature_dimensions[feature_type][0], img_dimensions[0]+1, feature_growth[feature_type][0]):
                for feature_height in range(min_feature_dimensions[feature_type][1], img_dimensions[1]+1, feature_growth[feature_type][1]):
                    for x_coord in range(img_dimensions[0]-feature_width+1):
                        for y_coord in range(img_dimensions[1]-feature_height+1):
                            top_left = [x_coord, y_coord]
                            bottom_right = [x_coord+feature_width, y_coord+feature_height]
                            self.all_features.append(HaarFeatures(feature_type, [top_left, bottom_right], [feature_width, feature_height], 1))
                            self.all_features.append(HaarFeatures(feature_type, [top_left, bottom_right], [feature_width, feature_height], -1))
        
    def Features(self):
        return self.all_features

def Driver():
    gf = GenerateFeatures([19,19])
    all_features = gf.Features()

#Driver()