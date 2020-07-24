import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn import linear_model

X, y = make_classification(n_samples=50000, n_features=15, n_informative=10, n_redundant=5,
                           n_classes=2, weights=[0.7], class_sep=0.7, random_state=15)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

def initialize_weights(dim):
    w = np.zeros_like(dim)
    b = 0
    return w,b

dim=X_train[0] 
w,b = initialize_weights(dim)

dim=X_train[0]
w,b = initialize_weights(dim)
def grader_weights(w,b):
    assert((len(w)==len(dim)) and b==0 and np.sum(w)==0.0)
    return True
grader_weights(w,b)


import math
def sigmoid(z):
    ''' In this function, we will return sigmoid of z'''
    # compute sigmoid(z) and return
    sigmoid = 1 / ( 1 + math.exp(-1 * z))
    return sigmoid

def grader_sigmoid(z):
    value =sigmoid(z)
    assert(value==0.8807970779778823)
    return True
grader_sigmoid(2)


def logloss(y_true,y_pred):
    '''In this function, we will compute log loss '''
    individual_loss = 0
    for index in range(len(y_true)):
        y_t = y_true[index]
        y_pr = y_pred[index]
        individual_loss += (y_t * math.log10(abs(y_pr))) + ((1 - y_t) * math.log10(abs(1 - y_pr) ))
    loss = (-1 / len(y_true)) * individual_loss
    return loss
def grader_logloss(true,pred):
    loss = logloss(true,pred)
    assert(loss==0.07644900402910389)
    return True
true=[1,1,0,1,0]
pred=[0.9,0.8,0.1,0.8,0.2]
grader_logloss(true,pred)


def gradient_dw(x,y,w,b,alpha,N):
    w_transpose = w.transpose()
    sigmoid_input = w_transpose.dot(x) + b
    sigmoid_output = sigmoid(sigmoid_input)
    #In this function, we will compute the gardient w.r.to w 
    dw  = x * ( (y - sigmoid_output) - (np.multiply(w,(alpha / N))))
    return dw

def grader_dw(x,y,w,b,alpha,N):
    grad_dww=gradient_dw(x,y,w,b,alpha,N)
    assert(np.sum(grad_dww)==2.613689585)
    return True

grad_x=np.array([-2.07864835,  3.31604252, -0.79104357, -3.87045546, -1.14783286,
       -2.81434437, -0.86771071, -0.04073287,  0.84827878,  1.99451725,
        3.67152472,  0.01451875,  2.01062888,  0.07373904, -5.54586092])

grad_y=0
grad_w,grad_b=initialize_weights(grad_x)
alpha=0.0001
N=len(X_train)
grader_dw(grad_x,grad_y,grad_w,grad_b,alpha,N)


 def gradient_db(x,y,w,b):
    '''In this function, we will compute gradient w.r.to b '''
    w_transform = w.transpose()
    sigmoid_input = w_transform.dot(x)
    sigmoid_output = sigmoid(sigmoid_input + b)
    db = y - sigmoid_output
    return db

def grader_db(x,y,w,b):
    grad_db=gradient_db(x,y,w,b)
    assert(grad_db==-0.5)
    return True

grad_x=np.array([-2.07864835,  3.31604252, -0.79104357, -3.87045546, -1.14783286,
       -2.81434437, -0.86771071, -0.04073287,  0.84827878,  1.99451725,
        3.67152472,  0.01451875,  2.01062888,  0.07373904, -5.54586092])
grad_y=0
grad_w,grad_b=initialize_weights(grad_x)
alpha=0.00016
N=len(X_train)
gradient_dbe(grad_x,grad_y,grad_w,grad_b)


from tqdm import tqdm
def train(X_train,y_train,X_test,y_test,epochs,alpha,eta0):
    ''' In this function, we will implement logistic regression'''
    #Here eta0 is learning rate
    #implement the code as follows
    # initalize the weights (call the initialize_weights(X_train[0]) function)
    grad_w, grad_b = initialize_weights(X_train[0])
    N=len(X_train)
    # for every epoch
    train_loss = []
    test_loss = []
    for each_epoc in range(epochs):
        # for every data point(X_train,y_train)
        for index in range(X_train.shape[0]):
            grad_x = X_train[index]
            #print(grad_x)
            grad_y = y_train[index]
            #compute gradient w.r.to w (call the gradient_dw() function)
            dw = gradient_dw(grad_x,grad_y,grad_w,grad_b,alpha,N)
            #compute gradient w.r.to b (call the gradient_db() function)
            db = gradient_db(grad_x,grad_y,grad_w,grad_b)
            #update w, b
            grad_w = grad_w + (alpha * dw)
            grad_b = grad_b + (alpha * db)
        y_train_pred_epoch = []
        for index in range(len(X_train)):
            y_train_pred_epoch.append((grad_w.transpose().dot(X_train[index])) + grad_b)
        train_loss.append(logloss(y_train,y_train_pred_epoch))

        y_test_pred_epoch = []
        for index in range(len(X_test)):
            y_test_pred_epoch.append((grad_w.transpose().dot(X_test[index])) + grad_b)
        test_loss.append(logloss(y_test,y_test_pred_epoch))


    return grad_w,grad_b, train_loss, test_loss

alpha=0.0001
eta0=0.0001
N=len(X_train)
epochs=50
w,b,train_loss, test_loss =train(X_train,y_train,X_test,y_test,epochs,alpha,eta0)


import matplotlib.pyplot as plt

def PlotLossCurve(epochs,train_loss,test_loss):
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, test_loss, label='Test Loss')

    plt.legend()
    plt.xlabel("Epocs")
    plt.ylabel("Loss")
    plt.title("ERROR PLOTS")
    plt.grid()
    plt.show()

PlotLossCurve(list(range(epochs)),train_loss, test_loss)

def pred(w,b, X):
    N = len(X)
    predict = []
    for i in range(N):
        z=np.dot(w,X[i])+b
        if sigmoid(z) >= 0.5: # sigmoid(w,x,b) returns 1/(1+exp(-(dot(x,w)+b)))
            predict.append(1)
        else:
            predict.append(0)
    return np.array(predict)
print(1-np.sum(y_train - pred(w,b,X_train))/len(X_train))
print(1-np.sum(y_test  - pred(w,b,X_test))/len(X_test))