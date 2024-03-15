# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Import numpy as np.

3.Give the header to the data.

4.Find the profit of population.

5.Plot the required graph for both for Gradient Descent Graph and Prediction Graph.

6.End the program.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: LUBINDHER S
RegisterNumber:212222240056
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate error
        errors=(predictions - y).reshape(-1,1)
        
        theta -= learning_rate*(1/len(X1)*X).T.dot(errors)
        
    return theta
data=pd.read_csv("C:/Users/SEC/Downloads/50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/DEEPAK2200233/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707676/cf905a24-d031-4d88-bfa0-7cce8278ad6a)

![image](https://github.com/DEEPAK2200233/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707676/b6137401-1fe4-4107-a078-9e8586b61dee)

![image](https://github.com/DEEPAK2200233/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707676/629d6516-e5b7-4e4f-94a8-f2574e952878)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
