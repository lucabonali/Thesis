import sns as sns
from keras import Sequential
from keras.layers import Dense
import pandas as pd
from datetime import datetime
from keras.callbacks import TensorBoard
from sklearn.manifold import TSNE
import seaborn as sns

def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

boat_csv = read_data("Data/Boat_data.csv")
boat_csv = boat_csv.drop(columns=["Unnamed: 0"])

now = datetime.now()
logdir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

tensorboard = TensorBoard(log_dir=logdir)




