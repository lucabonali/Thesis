from Old.Data_loader import Data_loader
from Old.Reconstruction_error_sequences import Reconstruction_error_sequences

data_loader = Data_loader()
nominal_data = data_loader.load_data(path="Data/Boat_data.csv", index_to_remove=39)
anomalous_data = data_loader.load_data(path="Data/Boat_data_curved.csv", index_to_remove=39)


trainX_nominal = data_loader.prepare_sequences(data=nominal_data,
                                               timesteps=150,
                                               interval=15)

trainX_anomalous = data_loader.prepare_sequences(data=anomalous_data,
                                                 timesteps=150,
                                                 interval=15)

# lstm_ae = LSTM_AE_Model(hidden_dimensions=10,
#                  intermediate_dimension=32,
#                  timesteps_sequences=150,
#                  input_dim=7,
#                  optimizer="adam",
#                  loss="mse",
#                  metrics=['accuracy'])
# lstm_ae.fit_model(x=trainX_nominal,
#                   y=trainX_nominal,
#                   batch_size=150,
#                   epochs=20)
# lstm_ae.model.save("Models/Nominal_LSTM.model")

print("GETTING MODEL AND PREDICTING")
model = data_loader.load_model("Models/Nominal_LSTM.model")
autoencoded_nominal = model.predict(trainX_nominal)
autoencoded_anomalous = model.predict(trainX_anomalous)

print("STARTING METHOD")


method = Reconstruction_error_sequences(window_size=150,
                                        interval=15,
                                        normal_data=trainX_nominal,
                                        autoencoded_nominal=autoencoded_nominal,
                                        anomalous_data=trainX_anomalous,
                                        autoencoded_anomalous=autoencoded_anomalous)

print("done")
