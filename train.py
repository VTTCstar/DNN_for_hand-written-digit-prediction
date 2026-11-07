import pandas as pd
import numpy as np
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("oddrationale/mnist-in-csv")

# print("Path to dataset files:", path)
def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def cost_func(self,expect_list):
        cost=0
        for i in range(len(self.act_list)):
            cost+=((self.act_list[i]-expect_list[i])**2)
        return cost
def sigmoid(z,beta):
    # return 1/(1+np.exp(-z))
    return z if z>0 else beta*z
def sigmoid_derivative(z,beta):
    return 1 if z>0 else beta
def derivatives_w(self,it,front_derivative_list,out,beta,front_list):#w_matric of i connects i,i-1 and it is i-1 layer,and fron_derivative_list is i+1 layer derivatives
    if front_derivative_list is None:
        a=np.zeros(len(self.act_list))
        b=np.zeros(len(self.act_list))
        w=np.zeros((self.w_matrix.shape))
        for i in range(len(self.bias_list)):
            a[i]=2*(self.act_list[i]-out[i])
            b[i]=a[i]*sigmoid_derivative(self.z_list[i],beta)
        # w=np.matrix((self.w_matrix.shape))
        for i in range(len(self.bias_list)):
            for j in range(len(it.bias_list)):
                w[i][j]=b[i]*it.act_list[j]   
        der_beta=None
        if self.number==1:
            der_beta=0
            for i in range(len(self.bias_list)):
                c=beta if self.act_list[i]<0 else 0
                der_beta+=(a[i]*c)
        return a,b,w,der_beta
    elif it is not None:
        a=np.zeros(len(self.act_list))
        b=np.zeros(len(self.act_list))
        for i in range(len(self.act_list)):
            for j in range(len(front_derivative_list)):
                a[i]+=front_derivative_list[j]*sigmoid_derivative(front_list.z_list[j],beta)*front_list.w_matrix[j][i]
            b[i]=a[i]*sigmoid_derivative(self.z_list[i],beta)
        w=np.zeros((self.w_matrix.shape))
        # for i in range(len(self.bias_list)):
        #     for j in range(len(it.bias_list)):
        #         w[i][j]=b[i]*it.act_list[j]
        w=np.outer(b,it.act_list)
        der_beta=None
        if self.number==1:
            der_beta=0
            for i in range(len(self.bias_list)):
                c=beta if self.act_list[i]<0 else 0
                der_beta+=(a[i]*c)
        return a,b,w,der_beta
    else:
        return None,None,None,None


class neuron_Layer:
    def __init__(self,bias_list,parent,w_matrix,number):
        self.act_list=None
        self.z_list=None
        self.bias_list=bias_list
        self.parent=parent
        self.w_matrix=w_matrix
        self.derivative_list=None
        self.number=number
        if(parent is None or parent.end==1):
            if parent is not None:
                parent.end=0
            self.end=1
        elif parent is not None:
            self.end=0
        else:
            self.end=1

def add_layer():
    n_list=[]
    prev=None
    n=int(input("Enter no of layers:"))
    for i in range(n):
        a=int(input(f"Enter no of neurons in {i}th layer:"))
        if i==0:
            m=neuron_Layer(np.zeros((a)),prev,None,i)
        else:
            m=neuron_Layer(np.zeros((a)),prev,np.random.normal(loc=0,scale=np.sqrt(2/(len(prev.bias_list))),size=(a,len(prev.bias_list))),i)
        
        n_list.append(m)
        prev=m
    return n_list

def sdg(n_list,alpha,data2,targets,beta):#data is one example in a dataset
    w_whole=[[] for _ in range(len(n_list)-1)]
    b_whole=[[] for _ in range(len(n_list)-1)]
    rows=data2.to_numpy()
    targets2=targets.to_numpy()
    targets2=one_hot(targets2)
    for data,out in zip(rows,targets2):
        n_list[0].act_list=data
        for i in range(1,len(n_list)):
            n_list[i].z_list=np.zeros(len(n_list[i].bias_list))
            n_list[i].act_list=np.zeros(len(n_list[i].bias_list))
            # for j in range(len(n_list[i].bias_list)):
            #     for k in range(len(n_list[i-1].bias_list)):
            #         n_list[i].z_list[j]+=(n_list[i].w_matrix[j][k])*(n_list[i-1].act_list[k])
            #     n_list[i].z_list[j]+=n_list[i].bias_list[j]
            #     n_list[i].act_list[j]=(sigmoid(n_list[i].z_list[j],beta))
            n_list[i].z_list = n_list[i].w_matrix @ n_list[i-1].act_list + n_list[i].bias_list
            n_list[i].act_list = np.where(n_list[i].z_list>0, n_list[i].z_list, beta*n_list[i].z_list)
        # cost=cost_func(n_list[len(n_list)-1],out)
        front_list=None
        for i in range(len(n_list)-1,0,-1):
            if i==len(n_list)-1:
                a,b,w,der_beta=derivatives_w(n_list[i],n_list[i-1],front_list,out,beta,None)
            else:
                a,b,w,der_beta=derivatives_w(n_list[i],n_list[i-1],front_list,out,beta,n_list[i+1])
            front_list=a
            b_whole[i-1].append(b)
            w_whole[i-1].append(w)
    ders_w=[np.mean(w_whole[i], axis=0) for i in range(len(n_list)-1)]
    ders_b=[np.mean(b_whole[i], axis=0) for i in range(len(n_list)-1)]
    for i in range(1,len(n_list)):
        n_list[i].w_matrix-=alpha*(ders_w[i-1])
        n_list[i].bias_list-=alpha*(ders_b[i-1])
    beta-=alpha*der_beta
    return beta

def validate(train_data,n_list,targets,beta):
    cost=0
    rows=train_data.to_numpy()
    targets2=targets.to_numpy()
    targets2=one_hot(targets2)
    for data,out in zip(rows,targets2):
        n_list[0].act_list=data
        for i in range(1,len(n_list)):
            n_list[i].z_list=np.zeros(len(n_list[i].bias_list))
            n_list[i].act_list=np.zeros(len(n_list[i].bias_list))
            for j in range(len(n_list[i].bias_list)):
                for k in range(len(n_list[i-1].bias_list)):
                    n_list[i].z_list[j]+=(n_list[i].w_matrix[j][k])*(n_list[i-1].act_list[k])
                n_list[i].z_list[j]+=n_list[i].bias_list[j]
                n_list[i].act_list[j]=(sigmoid(n_list[i].z_list[j],beta))
        cost+=cost_func(n_list[len(n_list)-1],out)
    return cost


def final_sdg(n_list,data2,targets,alpha,test_data,test_target,beta):
    inc=0
    prev_cost=validate(test_data,n_list,test_target,beta)
    cost=prev_cost
    print(f"Epoch {1}: Cost = {prev_cost:.6f}")
    j=1
    while(inc<5):
        j+=1
        perm = np.random.permutation(len(data2))
        data2 = data2.iloc[perm]
        targets = targets.iloc[perm]
        for i in range(0,len(data2),100):
            beta=sdg(n_list,alpha,data2[i:i+100],targets[i:i+100],beta)
        prev_cost=cost
        cost=validate(test_data,n_list,test_target,beta)
        print(f"Epoch {j}: Cost = {cost:.6f}")
        if(prev_cost<cost):
            inc+=1
        
data_test=pd.read_csv('mnist_test.csv')
data_train=pd.read_csv('mnist_train.csv')
n_list=add_layer()
final_sdg(n_list,data_train.iloc[:,1:],data_train.iloc[:,0],0.01,data_test.iloc[:,1:],data_test.iloc[:,0],0.01)








