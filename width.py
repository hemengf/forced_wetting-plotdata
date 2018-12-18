from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from error_boxes import make_error_boxes as meb
from matplotlib import rc
rc('text',usetex=True)
data92 = np.genfromtxt('data_92.csv', delimiter=',',names=True)
data65 = np.genfromtxt('data_65.csv',delimiter=',',names=True)
data214 = np.genfromtxt('data_214.csv',delimiter=',',names=True)
data572 = np.genfromtxt('data_572.csv',delimiter=',',names=True)
data512 = np.genfromtxt('data_512.csv',delimiter=',',names=True)
data285 = np.genfromtxt('data_285.csv',delimiter=',',names=True)
data162 = np.genfromtxt('data_162.csv',delimiter=',',names=True)
data68 = np.genfromtxt('data_68.csv',delimiter=',',names=True)
data31 = np.genfromtxt('data_31.csv',delimiter=',',names=True)
data153 = np.genfromtxt('data_153.csv',delimiter=',',names=True)
data26= np.genfromtxt('data_26.csv',delimiter=',',names=True)
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
rho26=  1.248e3
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
gamma26 =  64.797e-3
dataset = [data572,data512,data285,data214,data162,data153,data92,data68,data65,data31,data26]
rhoset = [rho572,rho512,rho285,rho214,rho162,rho153,rho92,rho68,rho65,rho31,rho26]
gammaset = [gamma572,gamma512,gamma285,gamma214,gamma162,gamma153,gamma92,gamma68,gamma65,gamma31,gamma26]
viscosityset = [572,512,285,214,162,153,92,68,65,31,26]

drg = zip(dataset,rhoset,gammaset,viscosityset)
fig,ax = plt.subplots(figsize=(5,5))
plt.subplots_adjust(bottom=0.2,left=0.2)

correction = 6.93-2.2
correction2= 18.59-4.73

cmap = plt.get_cmap('tab20')
markerlist = ['o','>','^','<','v','s','H','*','p','D','h']
resultarray = np.load('umaxresult.npy')
muarray = resultarray[0,:]
umaxarray= resultarray[1,:]
umaxerrarray= resultarray[2,:]
def plotfunction(expo1,expo2,c):
    for i,(data,rho,gamma,mu) in enumerate(drg[:]):
        if mu == 68:
            continue
        phi = ((np.pi/180)*(data['leftangle']+data['rightangle'])/2)
        height = (data['lefth']+data['righth'])/2
        hflat = (data['rightflat']+data['leftflat'])/2
        hmin = (data['rightmin']+data['leftmin'])/2
        h1max = np.copy(data['1stmax'])
        h2max = np.copy(data['2ndmax'])
        h3max = np.copy(data['3rdmax'])
        tapewid = np.copy(data['tapewid'])
        R = np.copy(data['R'])/2
        thickw = np.copy(data['w'])

        if mu == 31:
            hflat += correction
            hmin += correction
            pass
        u = np.copy(data['tapevelocity'])
        du = np.copy(data['velocityuncertainty'])
        etha = np.repeat(mu,len(u)) 

        u = u[~np.isnan(thickw)]
        if len(u) == 0: 
            continue
        etha = etha[~np.isnan(thickw)]
        phi = phi[~np.isnan(thickw)]
        du = du[~np.isnan(thickw)]
        height = height[~np.isnan(thickw)]
        tapewid = tapewid[~np.isnan(thickw)]
        R = R[~np.isnan(thickw)]
        thickw = thickw[~np.isnan(thickw)]

        thickw = thickw/tapewid
        height = height/tapewid
        #if i not in [0,2,4,6,8,9]:
        #    continue
            
        ax.errorbar(u,thickw*12.7,fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor='C0',xerr=0, yerr=0,label='%scP'%mu)#for legend
        #ax.errorbar(thickw*mu**expo1,-height/2,fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor='C0',xerr=0, yerr=0,label='%scP'%mu)#for legend

        #ax.legend(ncol=1,loc=3)
        #ax.set_xlim(0,950)
        #ax.set_ylim(0,0.8)

        #if mu==572 or mu==285 or mu==162 or mu==26:
        #    ax.annotate("",xy=(umaxarray[i],0.0),xytext=(umaxarray[i],-0.25),arrowprops=dict(arrowstyle="wedge",color=cmap(i),lw=3),zorder=0)
        #else:
        #    ax.annotate("",xy=(umaxarray[i],0.0),xytext=(umaxarray[i],0.25),arrowprops=dict(arrowstyle="wedge",color=cmap(i),lw=3),zorder=0)
        U = np.arange(np.min(u)*1e-3,np.max(u)*1e-3,0.001)
        etha = 18.27e-6 
        Ca = etha*U/gamma

#xexpo1slider = plt.axes([0.25,0.08,0.65,0.03])
#xexpo2slider = plt.axes([0.25,0.05,0.65,0.03])
#xcslider = plt.axes([0.25,0.02,0.65,0.03])
#expo1slider = Slider(xexpo1slider,'expo1',-3,3,valinit=0)
#expo2slider = Slider(xexpo2slider,'expo2',-3,3,valinit=0)
#cslider = Slider(xcslider,'c',-300,300,valinit=0)
#
#
plotfunction(0,0,23)

#def update(val):
#    expo1 = expo1slider.val
#    expo2 = expo2slider.val
#    c= cslider.val
#    ax.clear()
#    plotfunction(expo1,expo2,c)
#    #fig.canvas.draw_idle()
#expo1slider.on_changed(update)
#expo2slider.on_changed(update)
#cslider.on_changed(update)
plt.tick_params(labelsize=18,right=True, top=True)

#ax.set_xlim(0,950)
ax.set_ylim(0,8)
ax.set_xlabel('$U (mm/s)$',fontsize=24,labelpad=0)
ax.set_ylabel(r'$W (mm)$',fontsize=24,labelpad=0)
plt.show()


