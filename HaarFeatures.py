import IntegralImage as iimg

feature_map = {0:"2HOR", 1:"2VER", 2:"3HOR", 3:"3VER", 4:"4SQR"}

class HaarFeatures:

    #Initialize all feature attributes
    def __init__(self, feature_type, feature_position, feature_dimensions, polarity, threshold=0, feature_weight=1):
        self.feature_type = feature_type
        self.feature_position = feature_position
        self.feature_dimensions = feature_dimensions
        self.polarity = polarity
        self.threshold = threshold
        self.feature_weight = feature_weight

    #Get score for the current feature on the integral image
    def get_score(self, integral_image):
        if feature_map[self.feature_type] == '2HOR':
            top_left = self.feature_position[0]
            bottom_right = self.feature_position[1]
            middle_top = [top_left[0], (top_left[1]+bottom_right[1])//2]
            middle_bottom = [bottom_right[0], middle_top[1]]
            white = iimg.sum_region(top_left, middle_bottom, integral_image)
            black = iimg.sum_region(middle_top, bottom_right, integral_image)
            score = white-black
        elif feature_map[self.feature_type] == '2VER':
            top_left = self.feature_position[0]
            bottom_right = self.feature_position[1]
            middle_left = [(top_left[0]+bottom_right[0])//2, top_left[1]]
            middle_right = [middle_left[0], bottom_right[1]]
            white = iimg.sum_region(top_left, middle_right, integral_image)
            black = iimg.sum_region(middle_left, bottom_right, integral_image)
            score = white-black
        elif feature_map[self.feature_type] == '3HOR':
            top_left = self.feature_position[0]
            bottom_right = self.feature_position[1]
            temp = bottom_right[1]//3
            black_top_left = [top_left[0], temp]
            black_bottom_left = [bottom_right[0], temp]
            black_top_right = [top_left[0], (temp*2)]
            black_bottom_right = [bottom_right[0], (temp*2)]
            white1 = iimg.sum_region(top_left, black_bottom_left, integral_image)
            black = iimg.sum_region(black_top_left, black_bottom_right, integral_image)
            white2 = iimg.sum_region(black_top_right, bottom_right, integral_image)
            score = white1+white2-black
        elif feature_map[self.feature_type] == '3VER':
            top_left = self.feature_position[0]
            bottom_right = self.feature_position[1]
            temp = bottom_right[0]//3
            black_top_left = [temp,   top_left[1]]
            black_top_right = [temp, bottom_right[1]]
            black_bottom_left = [(temp * 2) , top_left[1]]
            black_bottom_right = [black_bottom_left[0], bottom_right[1]]
            white1 = iimg.sum_region(top_left, black_top_right, integral_image)
            black = iimg.sum_region(black_top_left, black_bottom_right, integral_image)
            white2 = iimg.sum_region(black_bottom_left, bottom_right, integral_image)
            score = white1+white2-black
        elif feature_map[self.feature_type] == '4SQR':
            top_left = self.feature_position[0]
            bottom_right = self.feature_position[1]
            middle = [(bottom_right[0]-top_left[0])//2, (bottom_right[1]-top_left[1])//2]
            top_middle = [top_left[0], middle[1]]
            left_middle = [middle[0], top_left[1]]
            bottom_middle = [bottom_right[0], middle[1]]
            right_middle = [middle[0], bottom_right[1]]
            white1 = iimg.sum_region(top_left, middle, integral_image)
            white2 = iimg.sum_region(middle, bottom_right, integral_image)
            black1 = iimg.sum_region(top_middle, right_middle, integral_image)
            black2 = iimg.sum_region(left_middle, bottom_middle, integral_image)
            score = white1+white2-black1-black2
        return int(score)

    #Get vote for the current feature on the integral image
    def get_vote(self, integral_image):   
        score_f = self.get_score(integral_image)
        if score_f < self.polarity * self.threshold:
            vote_f=1
        else:
            vote_f=-1
        return (self.feature_weight * vote_f)