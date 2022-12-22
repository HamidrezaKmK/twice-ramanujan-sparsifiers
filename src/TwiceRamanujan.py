import math
import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sympy

class TwiceRamanujan:
    def __init__(self,L,d):
        if type(L)==nx.classes.graph.Graph:
            self.L=nx.laplacian_matrix(L).toarray()
        else:
            self.L=L
        print(self.L)
        self.d=d
        self.n=self.L.shape[0]
        self.parameters()
        self.v=self.vectors(self.L)

    def parameters(self):
        d,n=self.d,self.n
        d_root=math.sqrt(d)
        self.delta_l=1
        self.delta_u=(d_root+1)/(d_root-1)
        epsilon_l=1/d_root
        epsilon_u=(d_root-1)/(d+d_root)
        self.l_0=-n/epsilon_l
        self.u_0=n/epsilon_u
        
    def vectors(self,L):
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
        v=scipy.linalg.sqrtm((scipy.linalg.pinv(L)))@b@np.diag(np.sqrt(weight))
        return v

    def L_Inverse(self):
        eigenValues,eigenVectors =scipy.linalg.eig(L)
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        InverseEigen=[0]+list(map(lambda x: 1/x,eigenValues[1:]))
        temp= np.matmul(eigenVectors,np.diag(InverseEigen))
        L_Inv=np.matmul(temp,eigenVectors.transpose())
        return L_Inv

    def L_SquareRoot(self):
        eigenValues,eigenVectors=scipy.linalg.eig(L)
        idx = eigenValues.argsort()   
        eigenValues = eigenValues[idx]
        eigenValues[0]=0
        eigenVectors = eigenVectors[:,idx]
        InverseEigen=[0]+list(map(lambda x: x**(1/2),eigenValues[1:]))
        temp= np.matmul(eigenVectors,np.diag(InverseEigen))
        L_Inv=np.matmul(temp,eigenVectors.transpose())
        return L_Inv


    def upper_bound_function(self,L,u,delta_u,v):
        n=self.L.shape[0]
        inv_diff=scipy.linalg.pinv((u+delta_u)*np.identity(n)-L-(u+delta_u)*(1/math.sqrt(n))*np.ones((n,n)))
        pot_diff=self.potential_upper(L,u+delta_u)-self.potential_upper(L,u)
        upper_bound=(v.T@(inv_diff@inv_diff)@v)/(-pot_diff)+v.T@(inv_diff)@v
        return upper_bound

    def lower_bound_function(self,L,l,delta_l,v):
        n=self.L.shape[0]
        inv_diff=scipy.linalg.pinv(L-(l+delta_l)*np.identity(n)+(l+delta_l)*(1/math.sqrt(n))*np.ones((n,n)))
        pot_diff=self.potential_lower(L,l+delta_l)-self.potential_lower(L,l)
        lower_bound=(v.T@(inv_diff@inv_diff)@v)/pot_diff-v.T@(inv_diff)@v
        return lower_bound

    def adjac(self,L):
        A=np.diag(np.diag(L))-L
        return A
        
    def potential_upper(self,L,u):
        eigenValues,eigenVectors=np.linalg.eig(L)
        upper=0
        for val in eigenValues:
            upper+=1/(u-val)
        upper-=1/u
        return upper

    def potential_lower(self,L,l):
        eigenValues,eigenVectors=np.linalg.eig(L)
        lower=0
        for val in eigenValues:
            lower+=1/(val-l)
        lower+=1/l
        return lower


    def sparsify(self):
        L,v,n,d=self.L,self.v,self.n,self.d
        l_0,u_0,delta_l,delta_u=self.l_0,self.u_0,self.delta_l,self.delta_u
        A=np.zeros((n,n))
        l,u=l_0,u_0
        for i in range(d*(n-1)):
            m=v.shape[1]
            l+=delta_l
            u+=delta_u
            for j in range(m):
                vj=v[:,j].reshape(-1,1)
                lb=self.lower_bound_function(A,l,delta_l,vj)
                ub=self.upper_bound_function(A,u,delta_u,vj)
                if(lb>ub):
                    s=2/(ub+lb)
                    A=A+s*vj@vj.T
                    print(str(i)+":"+str(j))
                    break
        L_s=scipy.linalg.sqrtm(self.L)@A@scipy.linalg.sqrtm(self.L)
        return np.around(L_s.real,5)
    
    def draw_graph(self,L):
        A=self.adjac(L)
        G=nx.from_numpy_matrix(A)
        A1=self.adjac(self.L)
        G2=nx.from_numpy_matrix(A1)
        pos2 = nx.spring_layout(G2)
        subax1 = plt.subplot(121)
        nx.draw(G2,pos2, with_labels=True, font_weight='bold')
        for edge in G2.edges(data='weight'):
            nx.draw_networkx_edges(G2, pos2,edgelist=[edge], width=0.1*edge[2])
        subax2 = plt.subplot(122)
        pos = nx.spring_layout(G)
        nx.draw(G,pos, with_labels=True, font_weight='bold')
        for edge in G.edges(data='weight'):
            nx.draw_networkx_edges(G, pos,edgelist=[edge], width=0.1*edge[2])
        plt.show()
    
    # def ellipse(self):
        # eigenValues,eigenVectors =scipy.linalg.eig(self.L)
        # idx = eigenValues.argsort()
        # eigenValues = eigenValues[idx]
        # eigenVectors = eigenVectors[:,idx]
        # eigen=list(map(lambda x: 1/x,eigenValues[1:]))
        # for i in range(len(eigen)):
            # globals()["x_"+str(i)] = sympy.symbols('x'+str(i))
        # var=np.array([globals()["x_"+str(i)] for i in range(len(eigen))]).reshape(-1,1)
        # expr=0
        # for i in range(len(eigen)):
            # expr+=eigen[i]*(var[i][0]**2)
        # eq1 = sympy.Eq(expr, 1)
        # x=[globals()["x_"+str(i)] for i in range(len(eigen))]
        # sol = sympy.solve([eq1], x,dict=True)
        # print(sol)
        # print(len(eigen))
        #sympy.calculus.util.continuous_domain(sol, x[1:], sympy.S.Reals)
        # p=None
        # for solu in sol:
            # if p is None:
                # p=sympy.plot(solu[x_0](-10,10),show=False)
            # else:
                # t=sympy.plot(solu[x_0],show=False)
                # p.extend(t)
        # p.show()
    # def ellipse_analytic(self):
        # globals()["x_"+str(i)] = sympy.symbols('x'+str(i))
        # var=np.array([[x],[y]])
        # temp
        # m=var.T@self.L@var
        # eq1 = Eq(m[0][0], 1)
        # sol = solve([eq1], [x, y])
        # for solu in sol:
            # plot(solu[0],solu[1])
        
class Clique:
    def __init__(self,n):
        self.__b__=self.__edges__(n)
        self.__L__=self.__laplacian__(self.__b__)

    def __edges__(self,n):
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

    def laplacian(self):
        return self.__L__
        
    def __laplacian__(self,b):
        return np.matmul(b,b.transpose())
            


if "__main__"==__name__:
    #L=Clique(3).laplacian()
    g=nx.barbell_graph(4,2)
    TR=TwiceRamanujan(g,d=2)
    L_s=TR.sparsify()
    print(L_s)
    #TR.ellipse()
    TR.draw_graph(L_s)
    
    