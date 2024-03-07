# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)

    for _ in range(num_iters):
    #calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
    #calculate erros
        error=(predictions-y).reshape(-1,1)
    #update thera using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(error) 
    
    return theta

data=pd.read_csv("/content/50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
### DATASET:
![image](https://github.com/Darshans05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115534676/c14bb4bb-4d61-4aac-838f-3620525852aa)
## VALUE OF X:
![image](https://github.com/Darshans05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115534676/8e17fa50-56f4-4426-95dd-db724d6e9848)
## VALUE OF X1_SCALED:
![image](https://github.com/Darshans05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115534676/ac3fd2a5-69eb-4f57-b822-d8c5de7863c2)
## PREDICTED VALUE:
![image](https://github.com/Darshans05/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/115534676/b2f95408-904e-48f1-88b1-1ebed0731a10)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
