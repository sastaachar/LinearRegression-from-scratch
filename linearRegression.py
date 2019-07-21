# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:43:10 2019

@author: admin
"""

#import pandas to import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#We can use this instead of the dot function 
def mx(x,slopes):
    mx_sum =0
    for i in range(len(slopes)):
        mx_sum += x[i]*slopes[i]
    return mx_sum

def R_sq_value(slopes, intercept, x_train, y_train):
    
    #change to int64 , cause float32 sometimes gives wrong answers due to overflow \(^o^)/
    #x_train = x_train.astype("int64")
    sum_of_errors = 0
    
    for i in range(len(x_train)):
        sum_of_errors += (y_train[i]  - ((np.dot(x_train[i,:],slopes)) + intercept))**2
        
    #avg  error
    sum_of_errors = sum_of_errors / len(x_train) 
   
    return sum_of_errors

def gradient_decent(x_train, y_train, slopes, intercept, LearningRate = 0.01):
   # print(slopes,intercept)
    x_gradients = np.zeros(x_train.shape[1])
    c_gradient = 0
    #print(x_gradients,c_gradient)
    for i in range(len(y_train)):     
        #yp is the predicted value
        yp = (np.dot(x_train[i,:],slopes)) + intercept
        print(yp)
        #print((np.dot(x_train[i,:],slopes)),x_train[i,:],slopes)
        #print("i "+str(i))
        for j in range(x_train.shape[1]):
            #print("j "+str(j))
            x_gradients[j] += -(x_train[i,j])*(y_train[i] - yp)
        c_gradient  += (-(y_train[i] - yp))
    
    x_gradients = x_gradients * (2/len(y_train))
    c_gradient = c_gradient * (2/len(y_train))
    
    print(x_gradients,c_gradient)
    updated_x_gradient = slopes - (LearningRate)*x_gradients
    updated_c_gradient = intercept - (LearningRate)*c_gradient
    
    #print(updated_x_gradient,updated_c_gradient)
    return [updated_x_gradient,updated_c_gradient]

    
def LinearRegression(x_train,   y_train,    LearningRate = 0.01, iteration = 10000):
   # print(x_train,y_train)
    # y = m1*x1 + m2*x2 + m3*x3 +....+ mn*xn + c  (n is the size of x_train i.e no of features), c is the intercept
    n = x_train.shape[1]
    intercept  = 0
    
    #These m1,m2....mn is stored in slopes , we will initialiae it with 0, then keep updating it
    slopes = np.zeros(n)
    #slopes = slopes.astype("int64")
    
    #Gradient Decent 
    
    #first we will define the loss(to measure how wrong we are i.e the error value)
    loss = R_sq_value(slopes, intercept, x_train, y_train)

    #we update the slopes using their Gradient values 
    #Gradient is a vector with the partial differential of the differnt 
    for i in range(iteration):
        #print("itr "+str(i))
        slopes, intercept = gradient_decent(x_train, y_train, slopes, intercept)
        print(R_sq_value(slopes, intercept, x_train, y_train))
        
    return [slopes,intercept]

def predict(x_test,slope,intercept):
    y_predict = np.zeros(len(x_test))
    for i in range(len(x_test)):
        y_predict[i] = np.dot((x_test[i,]),slope) + intercept
    return y_predict
    
def scale(x):

    for j in range(x.shape[1]):
        mean_x = 0
        for i in range(len(x)):
            mean_x += x[i,j]
        mean_x = mean_x / len(x)
        sum_of_sq = 0
        for i in range(len(x)):
            sum_of_sq += (x[i,j] - mean_x)**2
        stdev = sum_of_sq / (x.shape[0] -1)
        for i in range(len(x)):
            x[i,j] = (x[i,j] - mean_x) / stdev
    return x        
        
        
if __name__ == "__main__":
    
    #import the dataset
    data = pd.read_csv("sample Data.txt",delim_whitespace=True)
  
    y = data.iloc[:, -1].values
    x = data.iloc[:, 1:-1].values
    
    #split the data
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = scale(x_train)
    x_test = scale(x_test)
    slope,intercept  = LinearRegression(x_train,y_train) 
    
    y_predict = predict(x_test,slope,intercept)
