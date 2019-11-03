import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Reconstruction_error_sequences:

    def __init(self, window_size,
               interval,
               normal_data,
               autoencoded_nominal,
               anomalous_data,
               autoencoded_anomalous):

        self.normal_data = normal_data
        self.anomalous_data = anomalous_data
        self.anomalous_sequences = []
        self.anomaly_scores = []
        self.anomaly_scores_nominal = []
        self.reconstr_error_nominal_sequences = []
        self.reconstr_error_anomalous_sequences = []
        self.MUs = []
        self.SIGMAs = []
        self.window = window_size
        self.interval = interval

        self.calculate_error(autoencoded_values=autoencoded_nominal, nominal=True)

        self.calculate_error(autoencoded_values=autoencoded_anomalous, nominal=False)

        self.nominal_parameters()

        self.anomalies()

        self.plot_anomalies_scores()

    def calculate_error(self, autoencoded_values, nominal):
        cont = 0
        if nominal:
            for i in autoencoded_values:
                self.reconstr_error_nominal_sequences.append(abs(self.normal_data[cont:cont + self.window] - i))
                cont += self.interval
        else:
            for j in autoencoded_values:
                self.reconstr_error_anomalous_sequences.append(abs(self.anomalous_data[cont:cont + self.window] - j))
                cont += self.interval

    def nominal_parameters(self):
        self.MUs.append(np.average(self.reconstr_error_nominal_sequences[-1], axis=0))
        self.SIGMAs.append(np.cov(self.reconstr_error_nominal_sequences[-1], rowvar=False))

    def calculate_anomaly_score_sequence(self, i, element):
        scores = []
        weights = [1, 1, 1, 0.01, 0.01, 10, 10]
        for j, elem in enumerate(element):
            score = np.dot(np.transpose(element[j] - self.MUs[i]) * weights,
                           np.dot(self.SIGMAs[i],
                                  (element[j] - self.MUs[i]) * weights))
            scores.append(score)
        return np.average(scores)

    def anomalies(self):
        for i, element in enumerate(self.reconstr_error_anomalous_sequences):
            self.anomaly_scores.append(self.calculate_anomaly_score_sequence(i, element))

        for i, element in enumerate(self.reconstr_error_nominal_sequences):
            self.anomaly_scores_nominal.append(self.calculate_anomaly_score_sequence(i, element))

    def plot_anomalies_scores(self):
        plt.plot(self.anomaly_scores)
        plt.plot(self.anomaly_scores_nominal)
        plt.show()

    def detectOutliers(self, x, outlierConstant=20):
        a = np.array(x)
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        resultList = []
        outlierList = []

        list = a.tolist()
        for y in range(len(list)):
            if list[y] >= quartileSet[0] and list[y] <= quartileSet[1]:
                resultList.append(list[y])
            else:
                outlierList.append((y, list[y]))
                resultList.append(list[y - 1])
        return resultList, outlierList

    def get_outlier_lists(self):
        nominal_without_outliers, outlier_list = self.detectOutliers(x=self.anomaly_scores_nominal)

        anomalous_without_outliers, outliers_anomalous_list = self.detectOutliers(x=self.anomaly_scores)

        self.plot_anomalies_on_maps(outlier_list, outliers_anomalous_list)

    def plot_anomalies_on_maps(self, outlier_list, outlier_anomalous_list):
        boat_csv = pd.read_csv("Data/Boat_data.csv")
        boat_csv = boat_csv.drop(columns=["Unnamed: 0"])

        plt.plot(boat_csv["G_Lon"], boat_csv["G_Lat"])
        plt.title("Nominal anoamalies points LSTM_AE")
        for i in outlier_list:
            anomaly_position = i[0]*self.interval
            plt.plot(boat_csv["G_Lon"][anomaly_position:anomaly_position+self.window],
                     boat_csv["G_Lat"][anomaly_position:anomaly_position+self.window], 'bo')

        plt.show()

        an_csv = pd.read_csv("Data/Boat_data_curved.csv")
        an_csv = an_csv.drop(columns=["Unnamed: 0"])

        plt.plot(an_csv["G_Lon"], an_csv["G_Lat"])
        plt.title("Nominal anoamalies points LSTM_AE")
        for i in outlier_anomalous_list:
            anomaly_position = i[0] * self.interval
            plt.plot(an_csv["G_Lon"][anomaly_position:anomaly_position + self.window],
                     an_csv["G_Lat"][anomaly_position:anomaly_position + self.window], 'bo')

        plt.show()
#
#
# a = np.array(anomaly_scores_nominal)
# b = np.array(anomaly_scores)
# anom_scores = abs(a - b)
# plt.plot(anom_scores)
# plt.show()
