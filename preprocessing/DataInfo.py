import pandas as pd
from pathlib import Path

#==============================================================================

class DataInfo(object):
    """
        Class for representing a dataset, its protected and non-protected features
    """
    def __init__(self, datafile, index_list_protected_feature):
        self.datafile = datafile
        self.filename = Path(datafile).stem
        self.data = pd.read_csv(datafile)
        self.feature_labels  = self.data.columns
        self.num_features = self.feature_labels.size - 1     # last column is classification column
        self.protected_indexes = [i-1 for i in index_list_protected_feature]
        self.protected_labels = [self.feature_labels[i] for i in self.protected_indexes]
        self.non_protected_indexes = [i for i in range(self.num_features) if i not in self.protected_indexes]
        self.non_protected_labels = [self.feature_labels[i] for i in self.non_protected_indexes]



    def print_stats(self):
        print('Non-Protected Features: ')
        for f_label in self.non_protected_labels:
            print("\t" + f_label)
        print('Protected Features: ')
        for f_label in self.protected_labels:
            print("\t" + f_label)
        print('Classification Label: ')
        print("\t" + self.feature_labels[-1])

    def generate_protected_classification_csv(self, label):
        path_str = "results/" + self.filename + "/" + self.filename + "-" + label + ".csv"
        filepath = Path(path_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        headers = self.non_protected_labels
        headers.append(label)
        self.data.to_csv(filepath, columns = headers, index = False)

    def generate_all_protected_classification_csv(self):
        for label in self.protected_labels:
            self.generate_protected_classification_csv(label)

    
