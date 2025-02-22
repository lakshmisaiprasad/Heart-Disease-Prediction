# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pickle
loaded_model=pickle.load(open("C:/Users/b lakshmi sai prasad/Downloads/trained_model.sav", 'rb'))
input_data=(53,1,0,140,203,1,0,155,1,3.1,0,0,3)
# change the input data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)
# reshape the numpy array as we are predicting for only on instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
  print("The Person is not effected by Heart Disease");
else:
  print('The person is effected by Heart Disease')