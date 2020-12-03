from src.features.CountOfEventsFeature import CountOfEventsFeature
from src.features.base.Feature import Feature
from scipy.stats import entropy


class EntropyFeature(Feature):

    def __init__(self):
        self.name = 'Entropy'

    def get_result(self, window):
        count_of_event_feature = CountOfEventsFeature()
        # return entropy(count_of_event_feature.get_result(window))
        return 0