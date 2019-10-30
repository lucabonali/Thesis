import matplotlib.pyplot as plt
from keras import Sequential
import keras as kr
from keras.layers import Dense
import pandas as pd
from datetime import datetime
from keras.callbacks import TensorBoard
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def read_data(data_path):
    data = pd.read_csv(data_path)
    return data

boat_csv = read_data("Data/Anomalous_Boat_data.csv")
boat_csv = boat_csv.drop(columns=["Unnamed: 0"])

scaler = StandardScaler()
boat_csv = scaler.fit_transform(boat_csv)


# #### AUTOENCODER
# now = datetime.now()
# logdir = "tf_logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"
#
# tensorboard = TensorBoard(log_dir=logdir)
#
# model = Sequential()
# encoder = Sequential(name="encoder")
# encoder.add(Dense(100, input_dim=boat_csv.shape[1], activation='relu'))
# encoder.add(Dense(64, activation="relu"))
# encoder.add(Dense(10, activation='relu'))
#
# decoder = Sequential()
# decoder.add(Dense(10, activation='relu'))
# decoder.add(Dense(64, activation="relu"))
# decoder.add(Dense(boat_csv.shape[1], activation='relu'))
#
# model.add(encoder)
# model.add(decoder)
#
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.fit(boat_csv,boat_csv,verbose=1,epochs=15, validation_split=0.1,callbacks=[tensorboard])
# ### END AUTOENCODER
#
# model.save("Models/Boat_nominal.model")


model = kr.models.load_model("Models/Boat_nominal.model")


