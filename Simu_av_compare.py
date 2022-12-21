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
    parser.add_argument("--L0norm", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100000)
    parser.add_argument("--alpha_scale", type=float, default=0.01)
    parser.add_argument("--distri", type=str, default="nonuniformly")
    parser.add_argument("--initial", type=str, default="uniformly")
    parser.add_argument("--gamma", type=float,default=0.1)
    parser.add_argument("--uni_scale", type=float,default=1)
    parser.add_argument("--non_uni_scale", type=float,default=1)
    parser.add_argument("--non_uni_start", type=float,default=0.01)
    parser.add_argument("--multi", type=int,default=10)
    opt=parser.parse_args()
    # Initialization X
    N=opt.n
    d=opt.d
    X=np.random.normal(size=(N,d))*np.sqrt(opt.uni_scale)

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
    multi=opt.multi
    for i in range(multi):
        if i==0:
            trainloss_array_sgd,testloss_array_sgd=SGD(alpha,X,y,beta_star,opt.iters,gamma)
            trainloss_array_sde,testloss_array_sde=SDE(alpha,X,y,beta_star,opt.iters,gamma)
            trainloss_array_asde,testloss_array_asde=AccurateSDE(alpha,X,y,beta_star,opt.iters,gamma)
        else:
            train_array_sgd,test_array_sgd=SGD(alpha,X,y,beta_star,opt.iters,gamma)
            train_array_sde,test_array_sde=SGD(alpha,X,y,beta_star,opt.iters,gamma)
            train_array_asde,test_array_asde=SGD(alpha,X,y,beta_star,opt.iters,gamma)
            trainloss_array_sgd+=train_array_sgd
            testloss_array_sgd+=test_array_sgd
            trainloss_array_sde+=train_array_sde
            testloss_array_sde+=test_array_sde
            trainloss_array_asde+=train_array_asde
            testloss_array_asde+=test_array_asde
    trainloss_array_sgd=trainloss_array_sgd/multi
    trainloss_array_sde=trainloss_array_sde/multi
    trainloss_array_asde=trainloss_array_asde/multi


    if opt.distri=='nonuniformly':
        os_path_str='SDE_'+opt.distri+'_gamma_'+str(opt.gamma)+'_non_uni_scale_'+str(opt.non_uni_scale)+'_non_uni_start_'+str(opt.non_uni_start)+'_l0norm_'+str(opt.L0norm)
        if os.path.exists(os_path_str):
            pass
        else:
            os.mkdir(os_path_str)
    else:
        os_path_str='SDE_'+opt.distri+'_gamma_'+str(opt.gamma)+'_uni_scale_'+str(opt.uni_scale)+'_l0norm_'+str(opt.L0norm)
        if os.path.exists(os_path_str):
            pass
        else:
            os.mkdir(os_path_str)
    fig, ax = plt.subplots()
    ax.plot(x_axis[::10],np.log10(trainloss_array_sde)[::10])
    ax.plot(x_axis[::10],np.log10(trainloss_array_sgd)[::10])
    ax.plot(x_axis[::10],np.log10(trainloss_array_asde)[::10])

    plt.legend(['SDE','SGD','Accurate SDE'])
    plt.xlabel("log10-iter")
    plt.ylabel("log10-trainloss")
    plt.title("Train Loss")
    save_path=os.path.join(os_path_str,'Fig_12_repeat_'+str(multi)+'.jpg')
    plt.savefig(save_path)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_axis,np.log10(testloss_array_sde))
    ax.plot(x_axis,np.log10(testloss_array_sgd))
    ax.plot(x_axis,np.log10(testloss_array_asde))

    plt.legend(['SDE','SGD','Accurate SDE'])
    plt.xlabel("log10-iter")
    plt.ylabel("log10-testloss")
    plt.title("Test Loss")
    save_path=os.path.join(os_path_str,'Fig_13_repeat_'+str(multi)+'.jpg')
    plt.savefig(save_path)
    plt.show()














