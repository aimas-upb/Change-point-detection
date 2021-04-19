import numpy as np


class FeatureExtractor:
    def __init__(self, features):
        self.features = features

    def add_feature(self, feature):
        if feature not in self.features:
            self.features.append(feature)

    def remove_feature(self, feature):
        if feature in self.features:
            self.features.remove(feature)

    # this is the main method for building the feature vector from all features of a window
    def extract_features_from_window(self, window):
        result = []

        for feature in self.features:
            # gets the result of each feature in the window
            # each feature has a get_result method that gets the value computed by the given feature
            feature_result = feature.get_result(window)
            # print(feature.name + ' ' + str(feature_result))

            if isinstance(feature_result, list) or isinstance(feature_result, np.ndarray):
                result.extend(feature_result)
            else:
                result.append(feature_result)
        return result
