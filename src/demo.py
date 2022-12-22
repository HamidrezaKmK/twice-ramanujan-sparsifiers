import math
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
def L_Inverse(L):
    eigenValues,eigenVectors =scipy.linalg.eig(L)
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    InverseEigen=[0]+list(map(lambda x: 1/x,eigenValues[1:]))
    temp= np.matmul(eigenVectors,np.diag(InverseEigen))
    L_Inv=np.matmul(temp,eigenVectors.transpose())
    return L_Inv

def L_SquareRoot(L):
    eigenValues,eigenVectors=scipy.linalg.eig(L)
    #L_i=np.matmul(v.transpose)
    idx = eigenValues.argsort()   
    eigenValues = eigenValues[idx]
    eigenValues[0]=0
    eigenVectors = eigenVectors[:,idx]
    #print(eigenValues.shape)
    InverseEigen=[0]+list(map(lambda x: x**(1/2),eigenValues[1:]))
    temp= np.matmul(eigenVectors,np.diag(InverseEigen))
    L_Inv=np.matmul(temp,eigenVectors.transpose())
    return L_Inv
def lower_inverse(L):
    eigenValues,eigenVectors =np.linalg.eig(L)
    #L_i=np.matmul(v.transpose)
    idx = eigenValues.argsort()
    #print(eigenVectors)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    #print(eigenVectors)
    InverseEigen=[0]+list(map(lambda x: 1/x,eigenValues[1:]))
    temp= np.matmul(eigenVectors,np.diag(InverseEigen))
    L_Inv=np.matmul(temp,eigenVectors.transpose())
    return L_Inv
    
def upper_inverse(L):
    eigenValues,eigenVectors =np.linalg.eig(L)
    #L_i=np.matmul(v.transpose)
    idx = eigenValues.argsort()
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    InverseEigen=list(map(lambda x: 1/x,eigenValues[:-1]))+[0]
    temp= np.matmul(eigenVectors,np.diag(InverseEigen))
    L_Inv=np.matmul(temp,eigenVectors.transpose())
    return L_Inv

def upper_bound_function(L,u,delta_u,v):
    n=L.shape[0]
    inv_diff=scipy.linalg.pinv((u+delta_u)*np.identity(n)-L-(u+delta_u)*(1/math.sqrt(n))*np.ones((n,n)))
    pot_diff=potential_upper(L,u+delta_u)-potential_upper(L,u)
    upper_bound=(v.T@(inv_diff@inv_diff)@v)/(-pot_diff)+v.T@(inv_diff)@v
    
    return upper_bound
def lower_bound_function(L,l,delta_l,v):
    n=L.shape[0]
    inv_diff=scipy.linalg.pinv(L-(l+delta_l)*np.identity(n)+(l+delta_l)*(1/math.sqrt(n))*np.ones((n,n)))
    pot_diff=potential_lower(L,l+delta_l)-potential_lower(L,l)
    lower_bound=(v.T@(inv_diff@inv_diff)@v)/pot_diff-v.T@(inv_diff)@v
    return lower_bound
def adjac(L):
    A=np.diag(np.diag(L))-L
    return A
    
def potential_upper(L,u):
    eigenValues,eigenVectors=np.linalg.eig(L)
    upper=0
    for val in eigenValues:
        upper+=1/(u-val)
    upper-=1/u
    return upper

def potential_lower(L,l):
    eigenValues,eigenVectors=np.linalg.eig(L)
    #print(eigenValues)
    lower=0
    for val in eigenValues:
        lower+=1/(val-l)
    lower+=1/l
    return lower


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

def parameters(L,d):
    d_root=math.sqrt(d)
    n=L.shape[0]
    delta_l=1
    delta_u=(d_root+1)/(d_root-1)
    epsilon_l=1/d_root
    epsilon_u=(d_root-1)/(d+d_root)
    l_0=-n/epsilon_l
    u_0=n/epsilon_u
    return l_0,u_0,delta_l,delta_u

def vectors(L):
    n=L.shape[0]
    L_copy=L.copy()
    B=np.array([[]])
    weight=np.array([[]])
    b=None
    for i in range(n):
        for j in range(n):
            if (i==j):
                continue
            elif(L_copy[i][j]==0):
                continue
            else:
                if b is None:
                    b=np.zeros((n,1))
                    L_copy[j][i]=0
                    b[i][0]=1
                    b[j][0]=-1
                    weight=np.append(weight,[[-L_copy[i][j]]])
                    continue
                x=np.zeros((n,1))
                L_copy[j][i]=0
                x[i][0]=1
                x[j][0]=-1
                b=np.append(b,x,axis=1)
                weight=np.append(weight,[[-L_copy[i][j]]])
    #v=L_SquareRoot(L)@b@np.sqrt(weight)    #print(L_Inverse(L))
    v=scipy.linalg.sqrtm((scipy.linalg.pinv(L)))@b@np.diag(np.sqrt(weight))
    #print(v.shape)
    return v

def sparsify(L,d):
    n=L.shape[0]
    v=vectors(L)
    l_0,u_0,delta_l,delta_u=parameters(L,d)
    A=np.zeros((n,n))
    l=l_0
    u=u_0
    for i in range(d*(n-1)):
        m=v.shape[1]
        l+=delta_l
        u+=delta_u
        for j in range(m):
            vj=v[:,j].reshape(-1,1)
            #print(vj)
            lb=lower_bound_function(A,l,delta_l,vj)
            ub=upper_bound_function(A,u,delta_u,vj)
            if(lb>ub):
                s=2/(ub+lb)
                A=A+s*vj@vj.T
                print(str(i)+":"+str(j))
                break
    return A
def Laplacian(b):
    return np.matmul(b,b.transpose())
def draw_graph(L):
    A=adjac(L)
    G=nx.from_numpy_matrix(A)
    # A1=adjac(L_old)
    # G2=nx.from_numpy_matrix(A1)
    pos = nx.spring_layout(G)
    subax1 = plt.subplot(121)
    nx.draw(G,pos, with_labels=True, font_weight='bold')
    for edge in G.edges(data='weight'):
        nx.draw_networkx_edges(G, pos,edgelist=[edge], width=0.1*edge[2])
    plt.show()
if "__main__"==__name__:
    d=2
    b=(Edges(5))
    L=Laplacian(b)
    #draw_graph(L)
    L_h=scipy.linalg.sqrtm((scipy.linalg.pinv(L)))
    print(L_h@L@L_h@np.array([1,1,0,-1,-1]))
    L_s=sparsify(L,d)
    print(L_s)
    L_s=scipy.linalg.sqrtm(L)@L_s@scipy.linalg.sqrtm(L)
    np.set_printoptions(precision=1)
    print(L_s)
    draw_graph(np.around(L_s.real,5))
    
    