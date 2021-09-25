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

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
import numpy as np
#Question 1
T = np.zeros(2)
T[0] = 85 #Starting temp F
Ta = 65 # Ambient temp F
T[1] = 74 # First time step
dt = 2 #hours passes
K = -((T[1]-T[0])/(dt*(T[0]-Ta)))
print(K)
'''I saw that some people had 0.611, this is due to the -K(T-Ta) where they set T as final temperature. I set T as the inital
or starting temperature. According to class notes T is initial, but to this jyupter its current/final?'''
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
#Question 2
def kdiff (T1, T2, Ta, t):
    '''returns k when given the the inital, ending, and ambient temperature along with time elapsed.
         Arguments 
    ---------
    T1: inital temperature 
    T2: ending temperature
    Ta: ambient temperature
    t: time passed
    
    Returns
    -------
    k: material conductivity '''
    return (-((T2-T1)/(t*(T1-Ta))))
print(kdiff(85,74,65,2))
```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

t = np.linspace(0,2,50) #hours passes
dt = t[1]-t[0]

T = np.zeros(len(t))
T[0] = 85 #Starting temp F
Ta = 65 # Ambient temp F
k = 0.275

Tanal = Ta + (T[0]-Ta)*np.exp(-k*t)

for i in range(len(t)-1):
    T[i+1] = (-k*(T[i]-Ta)*dt+T[i])
    
plt.plot (t, Tanal,'o-',label='analytical')
plt.plot (t, T,'o-' ,label='numerical')
plt.legend();    
```

```{code-cell} ipython3
'''As t -> inf the final temeprature should reach ambient temperature and not any further'''
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

t = np.linspace(0,20,100) # 100 hours passes
dt = t[1]-t[0]

T = np.zeros(len(t))
T[0] = 98.6 #Starting temp F
Ta = 65 # Ambient temp F
k = 0.275

Tanal = Ta + (T[0]-Ta)*np.exp(-k*t)

#Euler Approx Model
for i in range(len(t)-1):
    T[i+1] = (-k*(T[i]-Ta)*dt+T[i])
    
plt.plot (t, Tanal,'o-',label='analytical')
plt.plot (t, T,'o-' ,label='numerical')
plt.legend();   
```

```{code-cell} ipython3
#Assuming that the ambient temperature is held at a constant temperature
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

T = np.zeros(2)
T[0] = 98.6 #Starting temp F
T[1] = 85 #ending temp F
Ta = 65 # Ambient temp F
k = 0.275

dt = (T[1]-T[0])/(-k*(T[1]-Ta))
t = [11-dt, 11]
print(t[0])
print('The body temperture was 98.6 when the time was around 8:30am')
plt.plot (t, T,'-',label='analytical')
plt.legend();  
```

4. Now that we have a working numerical model, we can look at the results if the
ambient temperature is not constant i.e. T_a=f(t). We can use the weather to improve our estimate for time of death. Consider the following Temperature for the day in question. 

    |time| Temp ($^o$F)|
    |---|---|
    |6am|50|
    |7am|51|
    |8am|55|
    |9am|60|
    |10am|65|
    |11am|70|
    |noon|75|
    |1pm|80|

    a. Create a function that returns the current temperature based upon the time (0 hours=11am, 65$^{o}$F) 
    *Plot the function $T_a$ vs time. Does it look correct? Is there a better way to get $T_a(t)$?

    b. Modify the Euler approximation solution to account for changes in temperature at each hour. 
    Compare the new nonlinear Euler approximation to the linear analytical model. 
    At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
def ambiTemp(t):
    '''returns ambient temperature (ta) given military time, not time passed
    ---------
    t = time in military
    
    Returns
    -------
    ta = ambient temperature '''
    if t >= 6 and t <= 7:
        return (50 + (t-6) )
    elif t > 7 and t < 8:
        return (51 + (t-7)*4 )
    elif t >= 8:
        return (55 + 5*(np.clip(t,None,13)-8))

import matplotlib.pyplot as plt 
x = np.linspace(6,13,50)

z = np.zeros(len(x))
for i in range(len(x)):
    z[i] = ambiTemp(x[i])
print(z)
plt.plot (x,z)

'''The original plot I got showed a step plot. This is because it is under the assumtion that the 
tempeterure stays the same through the hour and instantly changes once it hits a checkpoint. The final version
I went with a linear line between the points.

'''
```

```{code-cell} ipython3
t = np.linspace(6,13,50)
ta = (t-6)
dt = t[1]-t[0]
T = np.zeros(len(t))
k = 0.275
Ta = np.zeros(len(t))


#Solving Numerically
for i in range(len(t)-1): #Solves for 11AM +
    if t[i] <= 11: #Can't pinpoint where 11AM is if t doesn't have an 11, so sets closest point to 11AM as 85F
        T[i] = 85 #Temp at 11 AM 
        mark = i     #Marks where 11AM
    T[i+1] = (-k*(T[i]-ambiTemp(t[i]))*dt+T[i])
for i in range(mark): #solving for 6-11AM
    T[mark-i-1] = (k*ambiTemp(t[mark-i-1])-(T[mark-i]/dt))/(-1/dt+k)

#Solving analytically

    
```
