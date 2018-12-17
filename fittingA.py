from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import container

data= np.genfromtxt('fittingA.csv', delimiter=',',names=True)
mu_out = np.copy(data['mu'])*1e-3
mu_in = 18.27e-6*mu_out/mu_out 
gamma =  64.797e-3
A_exp = np.copy(data['A'])
dA_exp = np.copy(data['dA'])
log10Aerr = dA_exp/(A_exp*np.log(10))
R = mu_out/mu_in
r = mu_in/mu_out

fig,ax = plt.subplots(figsize=(5,5))
plt.subplots_adjust(bottom=0.2,left=0.2)
ax.errorbar(r,A_exp,fmt='o',ms=8,mew=1,ecolor='k',mfc='none',mec='k',xerr=0, yerr=dA_exp,capsize=4,label='Experiments')

rho = 1.235e3
phi = 2.7*np.pi/180
S = np.sin(phi)
C = np.cos(phi)
delta = phi-np.pi

RR = np.arange(R.min(),R.max())

rr = np.arange(r.min(),r.max(),1e-5)
A_scriven = (rr*(S-delta)+(phi-S))*(phi+S)*(S+delta)*(1-C)/(rr*(S*C-phi)*(delta**2-S**2)+(delta-S*C)*(phi**2-S**2))#1-alpha
A_scriven *= (2*18.27e-6/(rho*9.8))
A_scriven = np.power(A_scriven,0.5)
ax.plot(rr,A_scriven,'k--',label="Huh & Scriven \n ($\phi=2.7^{\circ}$)")


handles,labels=ax.get_legend_handles_labels()
new_handles=[]

for h in handles:
    if isinstance(h, container.ErrorbarContainer):
        new_handles.append(h[0])
    else:
        new_handles.append(h)
ax.legend(new_handles[::-1],labels[::-1],fontsize=15,loc=4)

ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0),useMathText=True)
ax.set_xlabel(r'$\eta_{in}/\eta_{out}$',fontsize=24,labelpad=0)
ax.set_ylabel(r'$A$',fontsize=24,labelpad=0)
plt.tick_params(labelsize=18,right=True,top=True)
plt.show()
