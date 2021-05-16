from scipy.stats import entropy

from src.features.base.Feature import Feature
from src.features.state.CountOfONEventsFeature import CountOfONEventsFeature


class EntropyONFeature(Feature):

    def __init__(self):
        self.name = 'Entropy based on ON motion sensors'

    def get_result(self, window):
        count_of_ON_event_feature = CountOfONEventsFeature()
        return entropy(count_of_ON_event_feature.get_result(window))