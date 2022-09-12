import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR

data = pd.read_csv('../../data/diabetes.csv')

"""
Cunfusion matrix

    |0  |  1  |  pred
0   |TN |  FN |
1   |FN |  TP | 
True|



Global metrics

Acuraccy

A = TN + TP / (TN + FN + Fn + TP)

F1-Macro
f1_macro = sum for all classes(corrects ones / all data * F1)

Local metrics

Precision
This works on the world of the prediction with the expected output
P0 = TN / TN+FN
P1 = TP/TP+FN

Recall
This works on the world of the real values for the classes (in the MLP)
R0 = TN / TN+FP
R1 = TP / TP+FN


Function F1 (equivalent to R2_score on regressors)

F1 = 2(P_i * R_i / P_i + R_1) armonic mean over precision and recall


Curvas ROC






"""