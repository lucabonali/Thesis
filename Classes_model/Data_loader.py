import Pandas as pd
from sklearn.preprocessing import StandardScaler


class DataLoader():
    def __init__(self, path, index_to_remove):
        self.path = path
        self.index_to_remove = index_to_remove

    def read_data(self):
        data = pd.read_csv(self.path)
        return data

    def load_data(self):
        csv = self.read_data()
        csv = csv.drop(columns=["Unnamed: 0"])
        csv = csv.drop(csv.index[:-self.index_to_remove])
        scaler = StandardScaler()
        data = scaler.fit_transform(csv)
        return data

