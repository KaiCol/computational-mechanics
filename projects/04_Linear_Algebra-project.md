---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# CompMech04-Linear Algebra Project
## Practical Linear Algebra for Finite Element Analysis

+++

In this project we will perform a linear-elastic finite element analysis (FEA) on a support structure made of 11 beams that are riveted in 7 locations to create a truss as shown in the image below. 

![Mesh image of truss](../images/mesh.png)

+++

The triangular truss shown above can be modeled using a [direct stiffness method [1]](https://en.wikipedia.org/wiki/Direct_stiffness_method), that is detailed in the [extra-FEA_material](./extra-FEA_material.ipynb) notebook. The end result of converting this structure to a FE model. Is that each joint, labeled $n~1-7$, short for _node 1-7_ can move in the x- and y-directions, but causes a force modeled with Hooke's law. Each beam labeled $el~1-11$, short for _element 1-11_, contributes to the stiffness of the structure. We have 14 equations where the sum of the components of forces = 0, represented by the equation

$\mathbf{F-Ku}=\mathbf{0}$

Where, $\mathbf{F}$ are externally applied forces, $\mathbf{u}$ are x- and y- displacements of nodes, and $\mathbf{K}$ is the stiffness matrix given in `fea_arrays.npz` as `K`, shown below

_note: the array shown is 1000x(`K`). You can use units of MPa (N/mm^2), N, and mm. The array `K` is in 1/mm_

$\mathbf{K}=EA*$

$  \left[ \begin{array}{cccccccccccccc}
 4.2 & 1.4 & -0.8 & -1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 1.4 & 2.5 & -1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -0.8 & -1.4 & 5.0 & 0.0 & -0.8 & 1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 -3.3 & 0.0 & -0.8 & 1.4 & 8.3 & 0.0 & -0.8 & -1.4 & -3.3 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 1.4 & -2.5 & 0.0 & 5.0 & -1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & -1.4 & 8.3 & 0.0 & -0.8 & 1.4 & -3.3 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 & 0.0 & 0.0 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & 1.4 & 8.3 & 0.0 & -0.8 & -1.4 & -3.3 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.4 & -2.5 & 0.0 & 5.0 & -1.4 & -2.5 & 0.0 & 0.0 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & -1.4 & 5.0 & 0.0 & -0.8 & 1.4 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -1.4 & -2.5 & 0.0 & 5.0 & 1.4 & -2.5 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & -3.3 & 0.0 & -0.8 & 1.4 & 4.2 & -1.4 \\
 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 0.0 & 1.4 & -2.5 & -1.4 & 2.5 \\
\end{array}\right]~\frac{1}{m}$

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

```{code-cell} ipython3
fea_arrays = np.load('./fea_arrays.npz')
K=fea_arrays['K']
K
```

In this project we are solving the problem, $\mathbf{F}=\mathbf{Ku}$, where $\mathbf{F}$ is measured in Newtons, $\mathbf{K}$ `=E*A*K` is the stiffness in N/mm, `E` is Young's modulus measured in MPa (N/mm^2), and `A` is the cross-sectional area of the beam measured in mm^2. 

There are three constraints on the motion of the joints:

i. node 1 displacement in the x-direction is 0 = `u[0]`

ii. node 1 displacement in the y-direction is 0 = `u[1]`

iii. node 7 displacement in the y-direction is 0 = `u[13]`

We can satisfy these constraints by leaving out the first, second, and last rows and columns from our linear algebra description.

+++

### 1. Calculate the condition of `K` and the condition of `K[2:13,2:13]`. 

a. What error would you expect when you solve for `u` in `K*u = F`? 

b. Why is the condition of `K` so large? __The problem is underconstrained. It describes stiffness of structure, but not the BC's. So, we end up with sumF=0 and -sumF=0__

c. What error would you expect when you solve for `u[2:13]` in `K[2:13,2:13]*u=F[2:13]`

```{code-cell} ipython3
print(np.linalg.cond(K))
print(np.linalg.cond(K[2:13,2:13]))

print('expected error in x=solve(K,b) is {}'.format(10**(16-16)))
print('expected error in x=solve(K[2:13,2:13],b) is {}'.format(10**(2-16)))
'''Problem is missing some BC thus the summation of forces is 0'''
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

def GaussNaive(A,y):
    '''GaussNaive: naive Gauss elimination
    x = GaussNaive(A,b): Gauss elimination without pivoting.
    solution method requires floating point numbers, 
    as such the dtype is changed to float
    
    Arguments:
    ----------
    A = coefficient matrix
    y = right hand side vector
    returns:
    ---------
    x = solution vector
    Aug = augmented matrix (used for back substitution)'''
    [m,n] = np.shape(A)
    Aug = np.block([A,y.reshape(n,1)])
    Aug = Aug.astype(float)
    if m!=n: error('Matrix A must be square')
    nb = n+1
    # Gauss Elimination 
    for k in range(0,n-1):
        for i in range(k+1,n):
            if Aug[i,k] != 0.0:
                factor = Aug[i,k]/Aug[k,k]
                Aug[i,:] = Aug[i,:] - factor*Aug[k,:]
    # Back substitution
    x=np.zeros(n)
    for k in range(n-1,-1,-1):
        x[k] = (Aug[k,-1] - Aug[k,k+1:n]@x[k+1:n])/Aug[k,k]
    return x,Aug
```

```{code-cell} ipython3
def LUNaive(A):
    '''LUNaive: naive LU decomposition
    L,U = LUNaive(A): LU decomposition without pivoting.
    solution method requires floating point numbers, 
    as such the dtype is changed to float
    
    Arguments:
    ----------
    A = coefficient matrix
    returns:
    ---------
    L = Lower triangular matrix
    U = Upper triangular matrix
    '''
    [m,n] = np.shape(A)
    if m!=n: error('Matrix A must be square')
    nb = n+1
    # Gauss Elimination
    U = A.astype(float)
    L = np.eye(n)

    for k in range(0,n-1):
        for i in range(k+1,n):
            if U[k,k] != 0.0:
                factor = U[i,k]/U[k,k]
                L[i,k]=factor
                U[i,:] = U[i,:] - factor*U[k,:]
    return L,U
```

```{code-cell} ipython3
def solveLU(L,U,b):
    '''solveLU: solve for x when LUx = b
    x = solveLU(L,U,b): solves for x given the lower and upper 
    triangular matrix storage
    uses forward substitution for 
    1. Ly = b
    then backward substitution for
    2. Ux = y
    
    Arguments:
    ----------
    L = Lower triangular matrix
    U = Upper triangular matrix
    b = output vector
    
    returns:
    ---------
    x = solution of LUx=b '''
    n=len(b)
    x=np.zeros(n)
    y=np.zeros(n)
        
    # forward substitution
    for k in range(0,n):
        y[k] = b[k] - L[k,0:k]@y[0:k]
    # backward substitution
    for k in range(n-1,-1,-1):
        x[k] = (y[k] - U[k,k+1:n]@x[k+1:n])/U[k,k]
    return x
```

```{code-cell} ipython3
def Kel(node1,node2):
    '''Kel(node1,node2) returns the diagonal and off-diagonal element stiffness matrices based upon
    initial angle of a beam element and its length the full element stiffness is
    K_el = np.block([[Ke1,Ke2],[Ke2,Ke1]])
    
    Out: [Ke1 Ke2]
         [Ke2 Ke1]   
    arguments:
    ----------
    node1: is the 1st node number and coordinates from the nodes array
    node2: is the 2nd node number and coordinates from the nodes array
    outputs:
    --------
    Ke1 : the diagonal matrix of the element stiffness
    Ke2 : the off-diagonal matrix of the element stiffness
    '''
    a = np.arctan2(node2[2]-node1[2],node2[1]-node1[1])
    l = np.sqrt((node2[2]-node1[2])**2+(node2[1]-node1[1])**2)
    Ke1 = 1/l*np.array([[np.cos(a)**2,np.cos(a)*np.sin(a)],[np.cos(a)*np.sin(a),np.sin(a)**2]])
    Ke2 = 1/l*np.array([[-np.cos(a)**2,-np.cos(a)*np.sin(a)],[-np.cos(a)*np.sin(a),-np.sin(a)**2]])
    return Ke1,Ke2
```

```{code-cell} ipython3
def f(s):
    plt.plot(r[ix],r[iy],'-',color=(0,0,0,1))
    plt.plot(r[ix]+u[ix]*s,r[iy]+u[iy]*s,'-',color=(1,0,0,1))
    #plt.quiver(r[ix],r[iy],u[ix],u[iy],color=(0,0,1,1),label='displacements')
    plt.quiver(r[ix],r[iy],F[ix],F[iy],color=(1,0,0,1),label='applied forces')
    plt.quiver(r[ix],r[iy],u[ix],u[iy],color=(0,0,1,1),label='displacements')
    plt.axis(l*np.array([-0.5,3.5,-0.5,2]))
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('Deformation scale = {:.1f}x'.format(s))
    plt.legend(bbox_to_anchor=(1,0.5))
```

### 2. Apply a 300-N downward force to the central top node (n 4)

a. Create the LU matrix for K[2:13,2:13]

b. Use cross-sectional area of $0.1~mm^2$ and steel and almuminum moduli, $E=200~GPa~and~E=70~GPa,$ respectively. Solve the forward and backward substitution methods for 

* $\mathbf{Ly}=\mathbf{F}\frac{1}{EA}$

* $\mathbf{Uu}=\mathbf{y}$

_your array `F` is zeros, except for `F[5]=-300`, to create a -300 N load at node 4._

c. Plug in the values for $\mathbf{u}$ into the full equation, $\mathbf{Ku}=\mathbf{F}$, to solve for the reaction forces

d. Create a plot of the undeformed and deformed structure with the displacements and forces plotted as vectors (via `quiver`). Your result for aluminum should match the following result from [extra-FEA_material](./extra-FEA_material.ipynb). _note: The scale factor is applied to displacements $\mathbf{u}$, not forces._

> __Note__: Look at the [extra FEA material](./extra-FEA_material). It
> has example code that you can plug in here to make these plots.
> Including background information and the source code for this plot
> below.


![Deformed structure with loads applied](../images/deformed_truss.png)

```{code-cell} ipython3
#Part A - Steel
fea_arrays = np.load('./fea_arrays.npz')
K=fea_arrays['K']
Es = 200*1000
A = .1
Ff = np.zeros(11)
Ff[5] = -300
l=300

nodes = np.array([[1,0,0],[2,0.5,3**0.5/2],[3,1,0],[4,1.5,3**0.5/2],[5,2,0],[6,2.5,3**0.5/2],[7,3,0]])
nodes[:,1:3]*=l
elems = np.array([[1,1,2],[2,2,3],[3,1,3],[4,2,4],[5,3,4],[6,3,5],[7,4,5],[8,4,6],[9,5,6],[10,5,7],[11,6,7]])
ix = 2*np.block([[np.arange(0,5)],[np.arange(1,6)],[np.arange(2,7)],[np.arange(0,5)]])
iy = ix+1

r = np.block([n[1:3] for n in nodes])
K=np.zeros((len(nodes)*2,len(nodes)*2))
for e in elems:
    ni = nodes[e[1]-1]
    nj = nodes[e[2]-1]

    Ke1,Ke2 = Kel(ni,nj)
    i1=int(ni[0])*2-2
    i2=int(ni[0])*2
    j1=int(nj[0])*2-2
    j2=int(nj[0])*2
    
    K[i1:i2,i1:i2]+=Ke1
    K[j1:j2,j1:j2]+=Ke1
    K[i1:i2,j1:j2]+=Ke2
    K[j1:j2,i1:i2]+=Ke2

#Solving for uf
           
uf = np.linalg.solve(Es*A*K[2:13,2:13],Ff)
u=np.zeros(2*len(nodes))
u[2:13]=uf

#Solving for F    
F=Es*A*K@u

xy={0:'x',1:'y'}
print('displacements:\n----------------')
for i in range(len(u)):
    print('u_{}{}:{:.2f} mm'.format(int(i/2)+1,xy[i%2],u[i]))
print('\nforces:\n----------------')
for i in range(len(F)):
    print('F_{}{}:{:.2f} N'.format(int(i/2)+1,xy[i%2],F[i]))
interact(f,s=5);
```

```{code-cell} ipython3
#Part A - Alum
fea_arrays = np.load('./fea_arrays.npz')
K=fea_arrays['K']
Ea = 70*1000
A = .1
Ff = np.zeros(11)
Ff[5] = -300
l=300

nodes = np.array([[1,0,0],[2,0.5,3**0.5/2],[3,1,0],[4,1.5,3**0.5/2],[5,2,0],[6,2.5,3**0.5/2],[7,3,0]])
nodes[:,1:3]*=l
elems = np.array([[1,1,2],[2,2,3],[3,1,3],[4,2,4],[5,3,4],[6,3,5],[7,4,5],[8,4,6],[9,5,6],[10,5,7],[11,6,7]])
ix = 2*np.block([[np.arange(0,5)],[np.arange(1,6)],[np.arange(2,7)],[np.arange(0,5)]])
iy = ix+1

r = np.block([n[1:3] for n in nodes])
K=np.zeros((len(nodes)*2,len(nodes)*2))
for e in elems:
    ni = nodes[e[1]-1]
    nj = nodes[e[2]-1]

    Ke1,Ke2 = Kel(ni,nj)
    i1=int(ni[0])*2-2
    i2=int(ni[0])*2
    j1=int(nj[0])*2-2
    j2=int(nj[0])*2
    
    K[i1:i2,i1:i2]+=Ke1
    K[j1:j2,j1:j2]+=Ke1
    K[i1:i2,j1:j2]+=Ke2
    K[j1:j2,i1:i2]+=Ke2

#Solving for uf
uf = np.linalg.solve(Ea*A*K[2:13,2:13],Ff)
u=np.zeros(2*len(nodes))
u[2:13]=uf

#Solving for F         
F=Ea*A*K@u

xy={0:'x',1:'y'}
print('displacements:')
for i in range(len(u)):
    print(f'u_{(int(i/2)+1)}{xy[i%2]}:{round(u[i],2)} mm')
print('\nforces:')
for i in range(len(F)):
    print('F_{}{}:{:.2f} N'.format(int(i/2)+1,xy[i%2],F[i]))
interact(f,s=2);
```

### 3. Determine cross-sectional area

a. Using aluminum, what is the minimum cross-sectional area to keep total y-deflections $<0.2~mm$?

b. Using steel, what is the minimum cross-sectional area to keep total y-deflections $<0.2~mm$?

c. What are the weights of the aluminum and steel trusses with the
chosen cross-sectional areas?

```{code-cell} ipython3
elem = 11
l = 300 #mm 

s_rho = 0.00805 #g/mm^3
a_rho = 0.00271 #g/mm^3

E_s = 200*1000#MPa
E_al = 70*1000#MPa
N = 10000
#Al
A_al = np.linspace(10,50,N) #mm^2
for i in A_al:
    F_al=np.zeros(11)
    F_al[5]=-300
    uf_al = np.linalg.solve(E_al*i*K[2:13,2:13],F_al)
    u_al=np.zeros(len(K[2:13,2:13]))
    u_al[2:13]=uf_al[2:13]

    F_al=E_al*i*K[2:13,2:13]@u_al
    xy={0:'x',1:'y'}
    max_u_al = np.max(np.abs(u_al))
    #print(f'Max_u_al is {max_u_al} for Area of {i}]')
    if max_u_al < 0.2:
        #print(max_u_al)
        min_A = i - 50/N
        truss_weight = a_rho*i*l*elem
        break
        
print(f'The minimum required cross sectional area is {round(min_A,4)} with a weight of {round(truss_weight,4)}g for AL')

#Steel
A_steel = np.linspace(1,20,N) #mm^2
for i in A_al:
    F_s=np.zeros(11)
    F_s[5]=-300
    uf_s = np.linalg.solve(E_s*i*K[2:13,2:13],F_s)
    u_s=np.zeros(len(K[2:13,2:13]))
    u_s[2:13]=uf_s[2:13]

    F_s=E_s*i*K[2:13,2:13]@u_s
    xy={0:'x',1:'y'}
    max_u_s = np.max(np.abs(u_s))
    if max_u_s < 0.2:
        #print(max_u_al)
        min_A = i - 50/N
        truss_weight = s_rho*i*l*elem
        break
print(f'The minimum required cross sectional area is {round(min_A,4)} with a weight of {round(truss_weight,4)}g for Steel')
```

## References

1. <https://en.wikipedia.org/wiki/Direct_stiffness_method>
