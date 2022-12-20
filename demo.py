import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def L_Inverse(L):
    eigenValues,eigenVectors =np.linalg.eig(L)
    L_i=np.matmul(v.transpose)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    InverseEigen=[0]+list(map(lambda x: 1/x,eigenValues[1:]))
    temp= np.matmul(eigenVectors,np.diag(InverseEigen))
    L_Inv=np.matmul(temp,eigenVectors.transpose())
    return L_Inv

def L_SquareRoot(L):
    eigenValues,=np.linalg.eig(L)
    L_i=np.matmul(v.transpose)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    InverseEigen=[0]+list(map(lambda x: x**(1/2),eigenValues[1:]))
    temp= np.matmul(eigenVectors,np.diag(InverseEigen))
    L_Inv=np.matmul(temp,eigenVectors.transpose())
    return L_Inv

def adjac(L):
    A=np.diag(np.diag(L))-L
    return A
    
def potential_upper(L,u):
    eigenValues=np.linalg.eig(L)
    for val in eigenValues:
        upper+=1/(u-val)
    upper-=1/u
    return upper

def potential_lower(L,l):
    eigenValues=np.linalg.eig(L)
    for val in eigenValues:
        lower+=1/(val-l)
    lower+=1/l
    
def Upper_Bound(L,v,delta):
    I=np.identity()
    np.linalg.inv((u+delta)*I-L)
    return upper

    
def Edges(n):
    p=np.identity(n)
    b=None
    for i in range(n):
        j=i+1
        while(j<n):
            edge=p[:,[i]]-p[:,[j]]
            if b is None:
                b=edge
                j+=1
                continue
            b=np.append(b,edge,axis=1)
            j+=1
    return b
 
def Laplacian(b):
    return np.matmul(b,b.transpose())

if "__main__"==__name__:
    b=(Edges(5))
    L=Laplacian(b)
    A=adjac(L)
    G=nx.from_numpy_matrix(A)
    subax1 = plt.subplot(121)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()