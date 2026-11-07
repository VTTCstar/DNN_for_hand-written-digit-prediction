import pandas as pd
import numpy as np

#one-hot encoding
def soft_max(z_matrix):
    z_matrix=z_matrix-z_matrix.max(axis=0,keepdims=True)
    exps=np.exp(z_matrix)
    return exps/exps.sum(axis=0)
def one_hot(expect_list):
    encoded=np.zeros((10,len(expect_list)))
    encoded[expect_list,np.arange(len(expect_list))]=1
    return encoded
def cost_func(outputs_matrix,expect_num_matrix):#cost can be found by one-hot encoding ecpect_num into a list and then using cross entropy loss
    z=np.sum(outputs_matrix*expect_num_matrix,axis=0)
    return np.mean(-np.log(np.clip(z,1e-12,None)))
def PReLU(z,beta):
    return z if z>0 else beta*z
def PReLU_der_z(z,beta):
    return 1 if z>0 else beta
def PReLU_der_beta(z):
    return 0 if z>0 else z

class neuron_Layer:
    def __init__(self,bias_list,parent,w_matrix):
        self.act_list=None
        self.z_list=None
        self.bias_list=bias_list
        self.parent=parent
        self.w_matrix=w_matrix
        self.beta_list=np.zeros(len(bias_list))


def derivatives_w(curr_act_matrix,prev_act_matrix,front_derivative_z_matrix,front,expect_matrix,self,curr_z_matrix,front_z_matrix):#w_matrix of i connects i,i-1  layers and it is i-1 layer,and front_derivative_list is i+1 layer derivatives
    if front_derivative_z_matrix is None:
        grad_b=curr_act_matrix-expect_matrix#a=derivative of loss function wrt pre-activations of the current layer
        grad_a=(-1/curr_act_matrix)*(expect_matrix)
        grad_w=(grad_b@prev_act_matrix.T)/ curr_act_matrix.shape[1]
        grad_b_matrix=grad_b
        grad_b=grad_b.mean(axis=1)
        return grad_a,grad_b,grad_w,None,grad_b_matrix
    else:
        front_beta_list=front.beta_list
        grad_a=front.w_matrix.T@(front_derivative_z_matrix)
        grad_b=grad_a*(curr_z_matrix>=0)+grad_a*(curr_z_matrix<0)*self.beta_list[:,None]
        grad_w=(grad_b@prev_act_matrix.T)/grad_b.shape[1]
        grad_beta=grad_a*(curr_z_matrix<0)*curr_z_matrix
        grad_b_matrix=grad_b
        grad_b=grad_b.mean(axis=1)
        grad_beta=grad_beta.mean(axis=1)
        return grad_a,grad_b,grad_w,grad_beta,grad_b_matrix

def add_layer():
    n_list=[]
    prev=None
    n=int(input("Enter no of layers:"))
    for i in range(n):
        a=int(input(f"Enter no of neurons in {i}th layer:"))
        if i==0:
            m=neuron_Layer(np.zeros((a)),prev,None)
        else:
            m=neuron_Layer(np.zeros((a)),prev,np.random.normal(loc=0,scale=np.sqrt(2/(len(prev.bias_list))),size=(a,len(prev.bias_list))))
        n_list.append(m)
        prev=m
    return n_list

def sdg(n_list,alpha,data2,targets):#data is one example in a dataset
    w_whole=[]
    b_whole=[]
    beta_whole=[]
    n=len(n_list)
    X=data2.T
    n_layer_act_matrices=[]
    n_layer_z_matrices=[]
    n_layer_act_matrices.append(X)
    n_layer_z_matrices.append(X)
    for i in range(1,n):
        Z = n_list[i].w_matrix @ X + n_list[i].bias_list[:,None]
        if(i!=n-1):
            X = Z*(Z>=0)+Z*(Z<0)*n_list[i].beta_list[:,None]#numpy broadcasts along last axis
        else:
            X=soft_max(Z)
        n_layer_act_matrices.append(X)
        n_layer_z_matrices.append(Z)
    for i in range(n-1,0,-1):
        if i==n-1:
            a,b,w,der_beta,grad_b_matrix=derivatives_w(n_layer_act_matrices[i],n_layer_act_matrices[i-1],None,None,targets,n_list[i],n_layer_z_matrices[i],None)
        else:
            a,b,w,der_beta,grad_b_matrix=derivatives_w(n_layer_act_matrices[i],n_layer_act_matrices[i-1],front_list,n_list[i+1],targets,n_list[i],n_layer_z_matrices[i],n_layer_z_matrices[i+1])
        front_list=grad_b_matrix
        b_whole.append(b)
        w_whole.append(w)
        beta_whole.append(der_beta)

    for i in range(n-1,0,-1):
        n_list[i].bias_list-=alpha*b_whole[n-1-i]
        n_list[i].w_matrix-=alpha*w_whole[n-1-i]
        if(i!=n-1):
          n_list[i].beta_list-=alpha*beta_whole[n-1-i]

def validate(test_data,n_list,targets):
    cost=0
    n=len(n_list)
    X=test_data.T
    for i in range(1,n):
        Z = n_list[i].w_matrix @ X + n_list[i].bias_list[:,None]
        if(i!=n-1):
            X = Z*(Z>=0)+Z*(Z<0)*n_list[i].beta_list[:,None]
        else:
            X=soft_max(Z)
    cost+=cost_func(X,targets)
    return cost

def final_sdg(n_list,data,targets,alpha,test_data,test_target):
    inc=0
    prev_cost=validate(test_data,n_list,test_target)
    cost=prev_cost
    print(f"Epoch {1}: Cost = {prev_cost:.6f}")
    j=1
    while(inc<5):
        j+=1
        perm = np.random.permutation(len(data))
        data = data[perm]
        targets = targets[:,perm]
        for i in range(0,len(data),100):
            sdg(n_list,alpha,data[i:min(i+100,len(data))],targets[:,i:min(i+100,len(data))])
        prev_cost=cost
        cost=validate(test_data,n_list,test_target)
        print(f"Epoch {j}: Cost = {cost:.6f}")
        if(prev_cost<cost):
            inc+=1
        else:
            inc=0

data_test=pd.read_csv('/content/sample_data/mnist_test.csv')
data_train=pd.read_csv('/content/sample_data/mnist_train.csv')
n_list=add_layer()
test_data=data_test.iloc[:,1:]
test_target=data_test.iloc[:,0]
train_data=data_train.iloc[:,1:]
train_target=data_train.iloc[:,0]
test_data=test_data.to_numpy()
test_target=test_target.to_numpy()
train_data=train_data.to_numpy()
train_target=train_target.to_numpy()
train_data=train_data/255.0
test_data=test_data/255.0
test_target=one_hot(test_target)
train_target=one_hot(train_target)
final_sdg(n_list,train_data,train_target,0.03,test_data,test_target)
print("training completed!!!")



