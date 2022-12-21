import numpy as np
import numpy.linalg as  LA
from utils import *


def GD(alpha,X,y,beta_star,iters,gamma):
    w_plus=alpha
    w_minus=alpha
    Trainloss_array=[]
    Testloss_array=[]

    for i in range(iters):
        # Compute Loss
        beta=np.square(w_plus)-np.square(w_minus)

        loss=compute_loss(beta,X,y)
        Trainloss_array.append(loss)
        testloss=compute_testloss(beta,beta_star)
        Testloss_array.append(testloss)

        # Update 
        gd_beta=gradient_beta(beta,X,y)
        gd_wplus=gd_beta*2*w_plus
        gd_wminus=-gd_beta*2*w_minus

        w_plus=w_plus-gamma*gd_wplus
        w_minus=w_minus-gamma*gd_wminus


    return np.array(Trainloss_array),np.array(Testloss_array)

def SGD(alpha,X,y,beta_star,iters,gamma):
    w_plus=alpha
    w_minus=alpha
    Trainloss_array=[]
    Testloss_array=[]

    for i in range(iters):
        # Compute Loss
        beta=np.square(w_plus)-np.square(w_minus)
        loss=compute_loss(beta,X,y)
        Trainloss_array.append(loss)
        testloss=compute_testloss(beta,beta_star)
        Testloss_array.append(testloss)

        # Update
        index_s=np.random.choice(np.arange(0,X.shape[0]),1) 
        X_s=X[index_s]
        y_s=y[index_s]
        # print(X_s.shape)
        sgd_beta=gradient_beta(beta,X_s,y_s)
        sgd_wplus=sgd_beta*2*w_plus
        sgd_wminus=sgd_beta*2*w_minus

        w_plus=w_plus-gamma*sgd_wplus
        w_minus=w_minus+gamma*sgd_wminus


    return np.array(Trainloss_array),np.array(Testloss_array)


def SDE(alpha,X,y,beta_star,iters,gamma):
    w_plus=alpha
    w_minus=alpha
    Trainloss_array=[]
    Testloss_array=[]

    for i in range(iters*10):
        beta=np.square(w_plus)-np.square(w_minus)
        loss=compute_loss(beta,X,y)
        testloss=compute_testloss(beta,beta_star)
        if i%10==0:
            Trainloss_array.append(loss)
            Testloss_array.append(testloss)
        
        h=gamma/10.
        Z=np.random.normal(size=(X.shape[0],1))
        gd_beta=gradient_beta(beta,X,y)
        gd_wplus=gd_beta*2*w_plus
        gd_wminus=-gd_beta*2*w_minus

        w_plus_temp=w_plus[:,0]
        w_plus_diag=np.diag(w_plus_temp)
        w_plus=w_plus-gd_wplus*h+2*np.sqrt(gamma/X.shape[0]*loss)*w_plus_diag@X.T@Z*np.sqrt(h)

        w_minus_temp=w_minus[:,0]
        w_minus_diag=np.diag(w_minus_temp)
        w_minus=w_minus-gd_wminus*h-2*np.sqrt(gamma/X.shape[0]*loss)*w_minus_diag@X.T@Z*np.sqrt(h)

    return np.array(Trainloss_array),np.array(Testloss_array)

def AccurateSDE(alpha,X,y,beta_star,iters,gamma):
    w_plus=alpha
    w_minus=alpha
    Trainloss_array=[]
    Testloss_array=[]

    for i in range(iters*10):
        beta=np.square(w_plus)-np.square(w_minus)
        loss=compute_loss(beta,X,y)
        loss_array=compute_loss_array(X,beta,beta_star)
        loss_array=loss_array.reshape(loss_array.shape[0])
        testloss=compute_testloss(beta,beta_star)
        if i%10==0:
            Trainloss_array.append(loss)
            Testloss_array.append(testloss)
        
        h=gamma/10.
        Z=np.random.normal(size=(X.shape[0],1))
        gd_beta=gradient_beta(beta,X,y)
        gd_wplus=gd_beta*2*w_plus
        gd_wminus=-gd_beta*2*w_minus

        loss_diag=np.diag(loss_array)
        # print("Loss Diag",loss_diag.shape)
        w_plus_temp=w_plus[:,0]
        w_plus_diag=np.diag(w_plus_temp)
        w_plus=w_plus-gd_wplus*h+2*np.sqrt(gamma/X.shape[0])*w_plus_diag@X.T@np.sqrt(loss_diag)@Z*np.sqrt(h)

        w_minus_temp=w_minus[:,0]
        w_minus_diag=np.diag(w_minus_temp)
        w_minus=w_minus-gd_wminus*h-2*np.sqrt(gamma/X.shape[0])*w_minus_diag@X.T@np.sqrt(loss_diag)@Z*np.sqrt(h)

    return np.array(Trainloss_array),np.array(Testloss_array)