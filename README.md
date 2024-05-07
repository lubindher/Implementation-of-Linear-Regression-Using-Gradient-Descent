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
**data.head()** <br>
<img height=10% width=99% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/b3ef5ab5-c8d3-42d3-86d5-29eea435dac9"><br><br>
**X values**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Y values** <br>
<img height=10% width=48% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/4fa96a4f-0a85-4307-b011-2ab04b73b9a9">&emsp;<img height=10% width=28% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/72810905-e103-4c76-ae8c-8a62f25cce8b"><br>
<br>
**X scaled**&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**Y scaled** <br>
<img height=10% width=48% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/9b3626af-2148-45d1-a822-a4019da4a3f5">&emsp;<img height=10% width=28% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/a26092ce-9f5e-47b7-97e2-636a1ffe9dc7"><br><br>
**Predicted Value**<br>
<img height=5% width=49% src="https://github.com/ROHITJAIND/EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118707073/5f807fd5-7777-40aa-9bb4-ac2508e9026e">


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
