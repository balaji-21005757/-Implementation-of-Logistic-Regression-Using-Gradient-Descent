# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: K.Balaji
RegisterNumber: 212221230011 
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data = np.loadtxt("ex2data1.txt",delimiter=',')
X = data[:, [0,1]]
y = data[:, 2]
X[:5]
y[:5]
plt.figure()
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()
def costFunction(theta, X,y):
  h = sigmoid(np.dot(X, theta))
  J = -(np.dot(y, np.log(h))+np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T,h-y) / X.shape[0]
  return J, grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)
def cost(theta, X, y):
  h = sigmoid(np.dot(X, theta))
  J = -(np.dot(y, np.log(h))+ np.dot(1-y,np.log(1-h))) / X.shape[0]
  return J
def gradient(theta, X,y):
  h = sigmoid(np.dot(X, theta))
  grad = np.dot(X.T, h-y) / X.shape[0]
  return grad
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train, y), method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
 plotDecisionBoundary(res.x, X, y)
 prob = sigmoid(np.dot(np.array([1, 45, 85]), res.x))
 print(prob)
 def predict(theta, X):
  X_train = np.hstack((np.ones((X.shape[0], 1)), X))
  prob = sigmoid(np.dot(X_train, theta))
  return (prob >=0.5).astype(int)
np.mean(predict(res.x, X) == y)
```

## Output:
### 1. Array Value of x
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/7235c278-effb-4466-b8d3-c4e7406b5ffc)
### 2. Array Value of y 
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/cef3b56f-bf80-451f-985a-5c8348eac071)
### 3. Exam 1 - score graph
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/60e67bb6-28cf-41d7-8115-315321da6bc4)
### 4. Sigmoid function graph
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/52df640e-caaa-4951-873f-d8f2f4d1a684)
### 5. X_train_grad value
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/ae219597-0f5f-4cb9-b71a-936f1cfed31a)
### 6. Y_train_grad value
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/9e1d5814-c4b3-411d-a2fb-436f20ab52fe)
### 7. Print res.x
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/fa10cd14-ab80-43ce-91af-22830f88ca0a)
### 8. Decision boundary - graph for exam score
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/8ae869c3-9613-46d0-80d9-5807fff57b01)
### 9. Proability value
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/5bcc83e9-5a8c-4271-b115-1b1f80a21e58)
### 10. Prediction value of mean
![image](https://github.com/balaji-21005757/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94372294/b8ce3d63-79c7-4b42-a70b-3ee12eb52cd2)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

