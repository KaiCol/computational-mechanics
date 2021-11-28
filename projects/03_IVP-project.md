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

# Initial Value Problems - Project

![Initial condition of firework with FBD and sum of momentum](../images/firework.png)

+++

You are going to end this module with a __bang__ by looking at the
flight path of a firework. Shown above is the initial condition of a
firework, the _Freedom Flyer_ in (a), its final height where it
detonates in (b), the applied forces in the __Free Body Diagram (FBD)__
in (c), and the __momentum__ of the firework $m\mathbf{v}$ and the
propellent $dm \mathbf{u}$ in (d). 

The resulting equation of motion is that the acceleration is
proportional to the speed of the propellent and the mass rate change
$\frac{dm}{dt}$ as such

$$\begin{equation}
m\frac{dv}{dt} = u\frac{dm}{dt} -mg - cv^2.~~~~~~~~(1)
\end{equation}$$

If you assume that the acceleration and the propellent momentum are much
greater than the forces of gravity and drag, then the equation is
simplified to the conservation of momentum. A further simplification is
that the speed of the propellant is constant, $u=constant$, then the
equation can be integrated to obtain an analytical rocket equation
solution of [Tsiolkovsky](https://www.math24.net/rocket-motion/) [1,2], 

$$\begin{equation}
m\frac{dv}{dt} = u\frac{dm}{dt}~~~~~(2.a)
\end{equation}$$

$$\begin{equation}
\frac{m_{f}}{m_{0}}=e^{-\Delta v / u},~~~~~(2.b) 
\end{equation}$$

where $m_f$ and $m_0$ are the mass at beginning and end of flight, $u$
is the speed of the propellent, and $\Delta v=v_{final}-v_{initial}$ is
the change in speed of the rocket from beginning to end of flight.
Equation 2.b only relates the final velocity to the change in mass and
propellent speed. When you integrate Eqn 2.a, you will have to compare
the velocity as a function of mass loss. 

Your first objective is to integrate a numerical model that converges to
equation (2.b), the Tsiolkovsky equation. Next, you will add drag and
gravity and compare the results _between equations (1) and (2)_.
Finally, you will vary the mass change rate to achieve the desired
detonation height.

+++

__1.__ Create a `simplerocket` function that returns the velocity, $v$,
the acceleration, $a$, and the mass rate change $\frac{dm}{dt}$, as a
function of the $state = [position,~velocity,~mass] = [y,~v,~m]$ using
eqn (2.a). Where the mass rate change $\frac{dm}{dt}$ and the propellent
speed $u$ are constants. The average velocity of gun powder propellent
used in firework rockets is $u=250$ m/s [3,4]. 

$\frac{d~state}{dt} = f(state)$

$\left[\begin{array}{c} v\\a\\ \frac{dm}{dt} \end{array}\right] = \left[\begin{array}{c} v\\ \frac{u}{m}\frac{dm}{dt} \\ \frac{dm}{dt} \end{array}\right]$

Use [an integration method](../module_03/03_Get_Oscillations) to
integrate the `simplerocket` function. Demonstrate that your solution
converges to equation (2.b) the Tsiolkovsky equation. Use an initial
state of y=0 m, v=0 m/s, and m=0.25 kg. 

Integrate the function until mass, $m_{f}=0.05~kg$, using a mass rate change of $\frac{dm}{dt}=0.05$ kg/s. 

> __Hint__: your integrated solution will have a current mass that you can
> use to create $\frac{m_{f}}{m_{0}}$ by dividing state[2]/(initial mass),
> then your plot of velocity(t) vs mass(t)/mass(0) should match
> Tsiolkovsky's
> 
> $\log\left(\frac{m_{f}}{m_{0}}\right) =
> \log\left(\frac{state[2]}{0.25~kg}\right) 
> = \frac{state[1]}{250~m/s} = \frac{-\Delta v+error}{u}$ 
> where $error$ is the difference between your integrated state variable
> and the Tsiolkovsky analytical value.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

```{code-cell} ipython3
def simplerocket(state,dmdt=0.05, u=250):
    '''Computes the right-hand side of the differential equation
    for the acceleration of a rocket, without drag or gravity, in SI units.
    
    Arguments
    ----------    
    state : array of three dependent variables [y v m]^T
    dmdt : mass rate change of rocket in kilograms/s default set to 0.05 kg/s
    u    : speed of propellent expelled (default is 250 m/s)
    
    Returns
    -------
    derivs: array of three derivatives [v (u/m*dmdt-g-c/mv^2) -dmdt]^T
    '''
    
    dstate = np.zeros(np.shape(state))
    # your work
    dstate[2] = -dmdt
    dstate[1] = u*dmdt/state[2]
    dstate[0] = state[1]
    return dstate
```

__2.__ You should have a converged solution for integrating `simplerocket`. Now, create a more relastic function, `rocket` that incorporates gravity and drag and returns the velocity, $v$, the acceleration, $a$, and the mass rate change $\frac{dm}{dt}$, as a function of the $state = [position,~velocity,~mass] = [y,~v,~m]$ using eqn (1). Where the mass rate change $\frac{dm}{dt}$ and the propellent speed $u$ are constants. The average velocity of gun powder propellent used in firework rockets is $u=250$ m/s [3,4]. 

$\frac{d~state}{dt} = f(state)$

$\left[\begin{array}{c} v\\a\\ \frac{dm}{dt} \end{array}\right] = 
\left[\begin{array}{c} v\\ \frac{u}{m}\frac{dm}{dt}-g-\frac{c}{m}v^2 \\ \frac{dm}{dt} \end{array}\right]$

Use [two integration methods](../notebooks/03_Get_Oscillations.ipynb) to integrate the `rocket` function, one explicit method and one implicit method. Demonstrate that the solutions converge to equation (2.b) the Tsiolkovsky equation. Use an initial state of y=0 m, v=0 m/s, and m=0.25 kg. 

Integrate the function until mass, $m_{f}=0.05~kg$, using a mass rate change of $\frac{dm}{dt}=0.05$ kg/s, . 

Compare solutions between the `simplerocket` and `rocket` integration, what is the height reached when the mass reaches $m_{f} = 0.05~kg?$

```{code-cell} ipython3
#inital prep
m0=0.25
mf=0.05
dm=0.05
u = 250 #m/s
t = np.linspace(0,(m0-mf)/dm,500)
dt=t[1]-t[0]
u = 250 #m/s
#Solving using Ideal Rocket Eq
m_step = np.linspace(mf, m0, 500)
lnm = (np.log(m_step/m0)) 
v = u*lnm
plt.plot(m_step/m0, v, color = "black", label = "Tsiolkovsky's")
#Solving using ODE solver fia scipy
from scipy.integrate import solve_ivp
num_sol = solve_ivp(lambda t,y :simplerocket(y), [0,t.max()], [0,0,m0], t_eval = t)
plt.plot(num_sol.y[2]/m0, num_sol.y[1], '--',color = 'orange', label = 'Numerical')

#Plotting to see convergence
plt.ylabel('v (m/s)')
plt.xlabel('m/m0')
plt.title('Prompt equation')
plt.legend()
print('The first plot is done according to the prompt, it converges but it looks really bad\
 the second graph was redone according to the source website with the ideal equation for rockets')
#Plotting second plot to see convergence
plt.figure()
plt.title('Source Equation')
m_step = np.linspace(m0, mf, 500)
lnm = (np.log(m_step/mf)) 
v = u*lnm
plt.plot(m_step/mf, v, color = "black", label = "Tsiolkovsky's")
plt.plot(m0/num_sol.y[2], num_sol.y[1], '--',color = 'orange', label = 'Numerical')
plt.ylabel('v (m/s)')
plt.xlabel('m/m0')
plt.legend();
```

```{code-cell} ipython3
def rocket(state,dmdt=0.05, u=250,c=0.18e-3):
    '''Computes the right-hand side of the differential equation
    for the acceleration of a rocket, with drag, in SI units.
    
    Arguments
    ----------    
    state : array of three dependent variables [y v m]^T
    dmdt : mass rate change of rocket in kilograms/s default set to 0.05 kg/s
    u    : speed of propellent expelled (default is 250 m/s)
    c : drag constant for a rocket set to 0.18e-3 kg/m
    Returns
    -------
    derivs: array of three derivatives [v (u/m*dmdt-g-c/mv^2) -dmdt]^T
    '''
    g=9.81
    dstate = np.zeros(np.shape(state))
    # your work
    dstate[2] = -dmdt
    dstate[1] = u*dmdt/state[2]- g -c * state[1]**2/state[2]
    dstate[0] = state[1]
    #print(f'The dmdt is {dmdt}')
    return dstate
```

```{code-cell} ipython3
#Solving using ODE solver via Scipy
num_sol = solve_ivp(lambda t,y :rocket(y), [0,t.max()], [0,0,m0], t_eval = t)
m_step = np.linspace(m0, mf, 500)
lnm = (np.log(m_step/mf)) 
v = u*lnm
plt.plot(m_step/mf, v, color = "black", label = "Tsiolkovsky's")
plt.plot(m0/num_sol.y[2], num_sol.y[1], '--',color = 'orange', label = 'Numerical')
plt.ylabel('v (m/s)')
plt.xlabel('m/m0')
plt.title('v vs m/mo')
plt.legend();
print('Rocket w/drag diverages from the ideal rocket equation')
```

```{code-cell} ipython3
#Solving Rocket and Simple Rocket
num_solr = solve_ivp(lambda t,y :rocket(y), [0,t.max()], [0,0,m0], t_eval = t)
plt.plot(num_solr.t, num_solr.y[0], '--',color = 'blue', label = 'rocket')

num_solsr = solve_ivp(lambda t,y :simplerocket(y), [0,t.max()], [0,0,m0], t_eval = t)
plt.plot(num_solsr.t, num_solsr.y[0], '--',color = 'red', label = 'simplerocket')
#Plotting to compare
plt.ylabel('Height (m)')
plt.xlabel('time (s)')
plt.title('Height vs time')
plt.legend();
print(f"The simple rocket max height is {round(np.max(num_solsr.y[0]),2)} m, \
the rocket max height is {round(np.max(num_solr.y[0]),2)} m")
```

__3.__ Solve for the mass change rate that results in detonation at a height of 300 meters. Create a function `f_dm` that returns the final height of the firework when it reaches $m_{f}=0.05~kg$. The inputs should be 

$f_{m}= f_{m}(\frac{dm}{dt},~parameters)$

where $\frac{dm}{dt}$ is the variable you are using to find a root and $parameters$ are the known values, `m0=0.25, c=0.18e-3, u=250`. When $f_{m}(\frac{dm}{dt}) = 0$, you have found the correct root. 

Plot the height as a function of time and use a star to denote detonation at the correct height with a `'*'`-marker

Approach the solution in two steps, use the incremental search
[`incsearch`](../module_03/04_Getting_to_the_root) with 5-10
sub-intervals _limit the number of times you call the
function_. Then, use the modified secant method to find the true root of
the function.

a. Use the incremental search to find the two closest mass change rates within the interval $\frac{dm}{dt}=0.05-0.4~kg/s.$

b. Use the modified secant method to find the root of the function $f_{m}$.

c. Plot your solution for the height as a function of time and indicate the detonation with a `*`-marker.

```{code-cell} ipython3
def f_dm(dmdt, m0 = 0.25, c = 0.18e-3, u = 250):
    ''' define a function f_dm(dmdt) that returns 
    height_desired-height_predicted[-1]
    here, the time span is based upon the value of dmdt
    
    arguments:
    ---------
    dmdt: the unknown mass change rate
    m0: the known initial mass
    c: the known drag in kg/m
    u: the known speed of the propellent
    
    returns:
    --------
    error: the difference between height_desired and height_predicted[-1]
        when f_dm(dmdt) = 0, the correct mass change rate was chosen
    '''
    # your work
    #Calls on the ROCKET function to intergrate
    g=9.8 #m/s
    t = np.linspace(0,(m0-mf)/dmdt,500)
    num_sol = solve_ivp(lambda t,y :rocket(y,dmdt), [0,t.max()], [0,0,m0], t_eval = t)
    error = 300-num_sol.y[0][-1]
    return error
```

```{code-cell} ipython3
#Some testing
#dmdt = 0.5
#t = np.linspace(0,(m0-mf)/dmdt,500)
#num_sol = solve_ivp(lambda t,y :rocket(y,dmdt), [0,t.max()], [0,0,m0], t_eval = t)
#print(f'Non Function h_max is {num_sol.y[0][-1]} dmdt is 0.5')
lower, upper = incsearchrocket(f_dm,0.05, 0.4, ns = 20)
print(f'The closest mass change rates when the sign changes is {round(lower[0],4)} \
 to {round(upper[0],4)}')
```

```{code-cell} ipython3
#Modified incsearch for my method
def incsearchrocket(func,xmin,xmax,ns=50):
    '''incsearch: incremental search root locator
    xb = incsearch(func,xmin,xmax,ns):
      finds brackets of x that contain sign changes
      of a function on an interval
    arguments:
    ---------
    func = name of function
    xmin, xmax = endpoints of interval
    ns = number of subintervals (default = 50)
    returns:
    ---------
    xb(k,1) is the lower bound of the kth sign change
    xb(k,2) is the upper bound of the kth sign change
    If no brackets found, xb = [].'''
    x = np.linspace(xmin,xmax,ns)
    f = np.zeros(ns)
    for i in range(ns):
        f[i] = func(x[i])
    sign_f = np.sign(f)
    delta_sign_f = sign_f[1:]-sign_f[0:-1]
    i_zeros = np.nonzero(delta_sign_f!=0)
    nb = len(i_zeros[0])
    xb = np.block([[ x[i_zeros[0]+1]],[x[i_zeros[0]] ]] )

    
    if nb==0:
      print('no brackets found\n')
      print('check interval or increase ns\n')
    else:
      print('number of brackets:  {}\n'.format(nb))
    return xb
```

```{code-cell} ipython3
def mod_secant(func,dx,x0,es=0.0001,maxit=50):
    '''mod_secant: Modified secant root location zeroes
    root,[fx,ea,iter]=mod_secant(func,dfunc,xr,es,maxit,p1,p2,...):
    uses modified secant method to find the root of func
    arguments:
    ----------
    func = name of function
    dx = perturbation fraction
    x0 = initial guess
    es = desired relative error (default = 0.0001 )
    maxit = maximum allowable iterations (default = 50)
    p1,p2,... = additional parameters used by function
    returns:
    --------
    root = real root
    fx = func evaluated at root
    ea = approximate relative error ( )
    iter = number of iterations'''

    iter = 0;
    xr=x0
    for iter in range(0,maxit):
        xrold = xr;
        dfunc=(func(xr+dx)-func(xr))/dx;
        xr = xr - func(xr)/dfunc;
        if xr != 0:
            ea = abs((xr - xrold)/xr) * 100;
        else:
            ea = abs((xr - xrold)/1) * 100;
        if ea <= es:
            break
    return xr,[func(xr),ea,iter]
```

```{code-cell} ipython3
#Finding massflow rate using mod_secant and close guess from incsearch
dmdt, out = mod_secant(f_dm,0.0001,0.08,es=0.000001 )
print(f'{round(dmdt,5)} kg/s is the necessary mass flow rate needed to reach 300m')
print('the solve took ',out[2],' iterations')
#Setting up to solve using new dmdt value
t = np.linspace(0,(m0-mf)/dmdt,500)
num_sol = solve_ivp(lambda t,y :rocket(y,dmdt), [0,t.max()], [0,0,m0], t_eval = t)
plt.plot(t,num_sol.y[0])
plt.plot(t[-1],num_sol.y[0][-1], '*', markersize = 20, label = 'BOOM')
plt.annotate('BOOM',(t[-1],num_sol.y[0][-1]));
plt.title('height vs time of the rocket')
plt.xlabel('time (s)')
plt.ylabel('height (m)');
```

## References

1. Math 24 _Rocket Motion_. <https://www.math24.net/rocket-motion/\>

2. Kasdin and Paley. _Engineering Dynamics_. [ch 6-Linear Momentum of a Multiparticle System pp234-235](https://www.jstor.org/stable/j.ctvcm4ggj.9) Princeton University Press 

3. <https://en.wikipedia.org/wiki/Specific_impulse>

4. <https://www.apogeerockets.com/Rocket_Motors/Estes_Motors/13mm_Motors/Estes_13mm_1_4A3-3T>

```{code-cell} ipython3

```
