import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import argparse
from optimize_algo import *
from utils import *
import os


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--L0norm", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--alpha_scale", type=float, default=0.01)
    parser.add_argument("--distri", type=str, default="uniformly")
    parser.add_argument("--initial", type=str, default="uniformly")
    parser.add_argument("--gamma", type=float,default=0.04)
    parser.add_argument("--uni_scale", type=float,default=1)
    parser.add_argument("--non_uni_scale", type=float,default=5)
    parser.add_argument("--non_uni_start", type=float,default=0.1)
    opt=parser.parse_args()
    # Initialization X
    N=opt.n
    d=opt.d
    X=np.random.normal(size=(N,d))*opt.uni_scale

    if opt.distri=='nonuniformly':
        print("Non Uniformly Gaussian Training Data")
        var_array=np.linspace(opt.non_uni_start,opt.non_uni_scale,N)
        for i in range(N):
            X[i]=np.random.normal(scale=var_array[i],size=(1,d))
    
    Cov_X=1./N*X.T@X
    eigs,_=LA.eig(Cov_X)
    lambda_max=np.max(eigs)
    print(lambda_max)

    # Initialization beta_star,y,Loss array
    sample_index=np.random.choice(np.arange(0,d),size=opt.L0norm)
    beta_star=np.zeros((d,1))
    for i in range(sample_index.shape[0]):
        beta_star[sample_index[i],0]=np.random.normal()
    # print(beta_star)

    y=X@beta_star
    # print(y.shape)

    beta=beta_star
    loss=compute_loss(beta,X,y)
    print(loss)

    # Define gamma, 
    gamma=opt.gamma
    x_axis=np.log10(np.arange(1,opt.iters+1))
    alpha=np.ones((d,1))*opt.alpha_scale
    if opt.initial=='nonuniformly':


        print("Non Uniformly Weights Gaussian Initialization")
        alpha=np.random.normal(size=(d,1))*opt.alpha_scale

    # Update: GD,SGD
    trainloss_array_gd,testloss_array_gd=GD(alpha,X,y,beta_star,opt.iters,gamma)
    trainloss_array_sgd,testloss_array_sgd=SGD(alpha,X,y,beta_star,opt.iters,gamma)
    trainloss_array_sde,testloss_array_sde=SDE(alpha,X,y,beta_star,opt.iters,gamma)

    loss_integral_sgd=compute_loss_integral(trainloss_array_sgd,gamma)

    alpha_temp=alpha[:,0]
    print("Alpha Temp",alpha_temp.shape)
    alpha_inf=np.diag(alpha_temp) @ np.exp(-2*gamma*loss_integral_sgd*np.diag(Cov_X))
    alpha_inf=alpha_inf.reshape(alpha_inf.shape[0],1)
    print("Alpha Inf",alpha_inf.shape)
    trainloss_array_gd_inf,testloss_array_gd_inf=GD(alpha_inf,X,y,beta_star,opt.iters,gamma)

    if opt.distri=='nonuniformly':
        os_path_str=opt.distri+'_gamma_'+str(opt.gamma)+'_non_uni_scale_'+str(opt.non_uni_scale)+'_non_uni_start_'+str(opt.non_uni_start)+'_l0norm_'+str(opt.L0norm)
        if os.path.exists(os_path_str):
            pass
        else:
            os.mkdir(os_path_str)
    else:
        os_path_str=opt.distri+'_gamma_'+str(opt.gamma)+'_uni_scale_'+str(opt.uni_scale)+'_l0norm_'+str(opt.L0norm)
        if os.path.exists(os_path_str):
            pass
        else:
            os.mkdir(os_path_str)
    fig, ax = plt.subplots()
    ax.plot(x_axis,np.log10(trainloss_array_sde))
    ax.plot(x_axis,np.log10(trainloss_array_sgd))
    ax.plot(x_axis,np.log10(trainloss_array_gd))

    plt.legend(['SDE','SGD','GD'])
    plt.xlabel("log10-iter")
    plt.ylabel("log10-trainloss")
    plt.title("Train Loss")
    save_path=os.path.join(os_path_str,'Fig_1.jpg')
    plt.savefig(save_path)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_axis,np.log10(testloss_array_sde))
    ax.plot(x_axis,np.log10(testloss_array_sgd))
    ax.plot(x_axis,np.log10(testloss_array_gd))

    plt.legend(['SDE','SGD','GD'])
    plt.xlabel("log10-iter")
    plt.ylabel("log10-testloss")
    plt.title("Test Loss")
    save_path=os.path.join(os_path_str,'Fig_2.jpg')
    plt.savefig(save_path)
    plt.show()


    fig, ax = plt.subplots()
    ax.plot(x_axis,np.log10(trainloss_array_sgd))
    ax.plot(x_axis,np.log10(trainloss_array_gd))
    ax.plot(x_axis,np.log10(trainloss_array_gd_inf))
    plt.legend(['SGD alpha','GD alpha','GD alpha_inf'])
    plt.xlabel("log10-iter")
    plt.ylabel("log10-trainloss")
    plt.title("Train Loss")
    save_path=os.path.join(os_path_str,'Fig_3.jpg')
    plt.savefig(save_path)
    plt.show()


    fig, ax = plt.subplots()
    ax.plot(x_axis,np.log10(testloss_array_sgd))
    ax.plot(x_axis,np.log10(testloss_array_gd))
    ax.plot(x_axis,np.log10(testloss_array_gd_inf))
    
    plt.legend(['SGD alpha','GD alpha','GD alpha_inf'])
    plt.xlabel("log10-iter")
    plt.ylabel("log10-testloss")
    plt.title("Test Loss")
    save_path=os.path.join(os_path_str,'Fig_4.jpg')
    plt.savefig(save_path)
    plt.show()











