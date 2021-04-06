class TransitionAccuracy:
    def __init__(self, main_activity, transition_activity, accuracy):
        self.main_activity = main_activity
        self.transition_activity = transition_activity
        self.accuracy = accuracy

    def to_string(self):
        return self.main_activity + " to " + self.transition_activity + " has accuracy " + str(self.accuracy)

