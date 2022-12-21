import numpy as np
import numpy.linalg as LA

def compute_loss_array(X,beta,beta_star):
    delta_beta=(beta-beta_star).reshape(-1,1)
    # print(delta_beta.shape)
    return 1./4.*(X@delta_beta)**2

def compute_loss(beta,X,y):
    return 1./4*np.mean((X@beta-y)**2)

def compute_testloss(beta,beta_star):
    return np.sum((beta-beta_star)**2)


def gradient_beta(beta,X,y):

    return 1./(2*X.shape[0])*X.T@(X@beta-y)


def compute_loss_integral(loss_array,gamma):
    loss_integral=0
    for loss in loss_array:
        loss_integral+=loss*gamma

    return loss_integral
