from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
correction = 6.93-2.2
data92 = np.genfromtxt('../2017_6_18_more/old_calibration/data.csv', delimiter=',',names=True)
data65 = np.genfromtxt('../2017_7_25_65cP/data.csv',delimiter=',',names=True)
data214 = np.genfromtxt('../2017_8_29_214cP/data.csv',delimiter=',',names=True)
data572 = np.genfromtxt('../2017_9_5_572cP/data.csv',delimiter=',',names=True)
data512 = np.genfromtxt('../2017_9_14_512cP/data.csv',delimiter=',',names=True)
data285 = np.genfromtxt('../2017_9_17_285cP/data.csv',delimiter=',',names=True)
data162 = np.genfromtxt('../2017_9_19_162cP/data.csv',delimiter=',',names=True)
data68 = np.genfromtxt('../2017_9_24_68cP/data.csv',delimiter=',',names=True)
data31 = np.genfromtxt('../2017_9_26_31cP/data.csv',delimiter=',',names=True)
data153 = np.genfromtxt('../2018_1_31_3colors_153cP/data.csv',delimiter=',',names=True)
data26= np.genfromtxt('../2018_2_21_26cP_3colors/data.csv',delimiter=',',names=True)
rho92 =  1.212e3
rho65 =  1.212e3
rho214 = 1.235e3
rho572 = 1.248e3
rho512 = 1.248e3
rho285 = 1.248e3
rho162 = 1.248e3
rho68 =  1.248e3
rho31=  1.248e3
rho153=  1.248e3
rho26= 1.248e3
gamma92 =  66.02e-3
gamma65 =  66.02e-3
gamma214 = 52.83e-3
gamma572 = 64.797e-3
gamma512 = 64.797e-3
gamma285 = 64.797e-3
gamma162 = 64.797e-3
gamma68 =  64.797e-3
gamma31 =  64.797e-3
gamma153 =  64.797e-3
gamma26  = 64.797e-3
dataset = [data572,data512,data285,data214,data162,data153,data92,data68,data65,data31,data26]
rhoset = [rho572,rho512,rho285,rho214,rho162,rho153,rho92,rho68,rho65,rho31,rho26]
gammaset = [gamma572,gamma512,gamma285,gamma214,gamma162,gamma153,gamma92,gamma68,gamma65,gamma31,gamma26]
viscosityset = [572,512,285,214,162,153,92,68,65,31,26]
drg = zip(dataset,rhoset,gammaset,viscosityset)
cmap = plt.get_cmap('tab20')
fig,ax = plt.subplots(figsize=(7,6))
plt.subplots_adjust(bottom=0.2,left=0.2)

thickwidth=[]
dthickwidth=[]
markerlist = ['o','>','^','<','v','s','H','*','p','D','h']
for i,(data,rho,gamma,mu) in enumerate(drg[:]):
    if mu == 68:
        continue
#for i,(data,rho,gamma,mu) in enumerate([drg[3]]):
    #if mu in [31,65,68]:
    #    continue
    plt.sca(ax)
    phi = ((np.pi/180)*(data['leftangle']+data['rightangle'])/2)
    height = (data['lefth']+data['righth'])/2
    hflat = (data['rightflat']+data['leftflat'])/2
    hmin = (data['rightmin']+data['leftmin'])/2
    R = np.copy(data['R'])/2
    w = np.copy(data['tapewid'])
    thickw = np.copy(data['w'])
    if mu == 31:
        hflat += correction
        hmin += correction
    u = np.copy(data['tapevelocity'])
    du = np.copy(data['velocityuncertainty'])
    etha = np.repeat(mu,len(u)) 

    #ignore np.nan values in the arrays
    #only for fitting purpose in file powerlawfitting.py
    u = u[~np.isnan(R)]
    if len(u) == 0:
        continue
    du = du[~np.isnan(R)]
    phi = phi[~np.isnan(R)]
    hmin = hmin[~np.isnan(R)]
    thickw = thickw[~np.isnan(R)]
    tapewid = w[~np.isnan(R)]
    thickw = thickw/tapewid
    R = R[~np.isnan(R)]
    normR = R/tapewid
    utan = u*np.sin(phi)
    #ax.errorbar(np.power(u*mu,0.5),normR,fmt='s',mfc='none',mec=cmap(i/float(20)),ecolor=cmap(i/float(20)),label='%scp'%mu)
    thickwidth.append(np.mean(normR))
    dthickwidth.append(np.std(normR))
    ax.errorbar(mu,np.mean(normR)*12.7,fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor="C0",yerr=np.std(normR)*12.7,capsize=4,label='%scp'%mu)
    ax.legend()
#ax.errorbar(viscosityset, thickwidth,fmt='o',ms=8,mfc='none',yerr=dthickwidth)
ax.tick_params(labelsize=12)
ax.set_ylim(0,9)
ax.set_xlabel('Liquid viscosity (cP)')
ax.set_ylabel('$R$ (mm)')
lcap = np.sqrt(66e-3/(1.25e4))*1e3
plt.axhline(y=lcap)
plt.show()
