import os

class TrainTestSplit:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def class_labels(self, age):
        if 1 <= age <= 2:
            return 0
        elif 3 <= age <= 9:
            return 1
        elif 10 <= age <= 20:
            return 2
        elif 21 <= age <= 27:
            return 3
        elif 28 <= age <= 45:
            return 4
        elif 46 <= age <= 65:
            return 5
        else:
            return 6
