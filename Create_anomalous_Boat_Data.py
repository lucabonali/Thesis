import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from math import degrees
from math import sqrt
from math import cos
from math import sin
from math import radians
import random


def headingFromCoordinates(lat1, lon1, lat2, lon2):
    if(lon2-lon1 < 0): #2nd and 3rd quadrant
        return 180 - degrees(np.arcsin((lat2-lat1)/(sqrt((lat2-lat1)**2 + (lon2-lon1)**2))))
    if(lat2-lat1 < 0): #4th quadrant
        return 360 + degrees(np.arcsin((lat2-lat1)/(sqrt((lat2-lat1)**2 + (lon2-lon1)**2))))
    return degrees(np.arcsin((lat2-lat1)/(sqrt((lat2-lat1)**2 + (lon2-lon1)**2)))) #1st quadrant

df = pd.read_csv('Data/Raw_boat_data')
df = df[210:6830]

h = [0]

for i in range(211, 6830):
    x = headingFromCoordinates(df['latitude'][i-1], df['longitude'][i-1], df['latitude'][i], df['longitude'][i])
    x = x + 140
    if(x > 360):
        x = x -360
    h.append(x)
df['degrees'] = h

df = df.drop(columns=["do","ec","temp","voltage"])
print(df.columns)
print(len(df))
df.to_csv("Data/Anomalous_boat_data.csv")

