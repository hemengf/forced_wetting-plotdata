from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from error_boxes import make_error_boxes as meb
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
gammaset = [gamma572,gamma512,gamma285,gamma214,gamma162, gamma153,gamma92,gamma68,gamma65,gamma31,gamma26]
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
def plotfunction(expolist):
    for i,(data,rho,gamma,mu) in enumerate(drg[:]):
        if mu == 68:
            continue
        #print mu
        plt.sca(ax)
        phi = ((np.pi/180)*(data['leftangle']+data['rightangle'])/2)
        tapewid=np.copy(data['tapewid'])
        heightleft=np.copy(data['lefth'])
        heightright=np.copy(data['righth'])
        height = (heightleft+heightright)/(2*tapewid)
        hflat = (data['rightflat']+data['leftflat'])/2
        hmin = (data['rightmin']+data['leftmin'])/2
        thickw = np.copy(data['w'])/tapewid

        h1max = np.copy(data['1stmax'])
        h2max = np.copy(data['2ndmax'])
        h3max = np.copy(data['3rdmax'])
        W = np.copy(data['VW'])/tapewid
        L = np.copy(data['L'])/tapewid
        R = np.copy(data['R'])/(2*tapewid)


        if mu == 31:
            hflat += correction
            hmin += correction
            h1max += correction
            h2max += correction
            h3max += correction
            pass
        u = np.copy(data['tapevelocity'])
        du = np.copy(data['velocityuncertainty'])
        etha = np.repeat(mu,len(u)) 
        #ignore np.nan values in h1max array
        """
        u = u[~np.isnan(height)]
        if len(u) == 0: 
            continue
        du = du[~np.isnan(height)]
        tapewid = tapewid[~np.isnan(height)]
        R = R[~np.isnan(height)]/tapewid
        thickw = thickw[~np.isnan(height)]/tapewid
        phi = phi[~np.isnan(height)]
        heightleft= heightleft[~np.isnan(height)]/tapewid
        heightright= heightright[~np.isnan(height)]/tapewid
        height= height[~np.isnan(height)]/tapewid
        """

        #if i not in [0,2,4,6,8,10]:
        #    continue

        #w = tapewid*447/627
        #heighttheory=0.5*np.sin(phi)*(w/np.tan(phi)+2*R-np.sqrt((w/np.tan(phi)+2*R)**2-(w/np.sin(phi))**2))/tapewid
        yerr = abs(heightleft-heightright)/2
        #viscosityset = [572,512,285,214,162,153,92,68,65,31,26]
        if mu==31 or mu==26:
            ax.errorbar(u-umaxarray[i], 12.7*height/expolist[i], fmt=markerlist[i], ms=8,mew=1,mfc=cmap(i),ecolor='none',xerr=0*du,yerr=0*yerr,capsize=4,label='%scP'%mu,zorder=0, alpha=0.3)#for legend
        else:
            ax.errorbar(u-umaxarray[i], 12.7*height/expolist[i], fmt=markerlist[i], ms=8,mew=1,mfc=cmap(i),ecolor='none',xerr=0*du,yerr=0*yerr,capsize=4,label='%scP'%mu,zorder=0, alpha=1)#for legend
        #ax.errorbar(np.power(u-umaxarray[i],expo1), height*np.power(mu,expo2), fmt=markerlist[i], ms=8,mew=1,mfc=cmap(i),ecolor='none',xerr=0*du,yerr=0*yerr,capsize=4,label='%scP'%mu,zorder=0)#for legend
        #ax.errorbar(thickw/2,-height,fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor='none',xerr=0,yerr=0,capsize=4,label='%scP'%mu,zorder=0)#for legend
        #ax.errorbar(u,R,fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor='none',xerr=0,yerr=0,capsize=4,label='%scP'%mu,zorder=0)#for legend
        #ax.axvline(umaxarray[i],color=cmap(i))
        #ax.legend(ncol=2)
        U = np.arange(np.min(u)*1e-3,np.max(u)*1e-3,0.001)
        etha = 18.27e-6 
        Ca = etha*U/gamma
        #ax.set_xlim(0,950)
        #ax.set_ylim(0,1)
        #ax.set_aspect('equal')
        #ax.set_xlim(0,0.4)
        #ax.set_ylim(-1,0)

coeffarray = [1,1.1,1.9,2,3,3,7.5,1,17,38,40]
coeffarray = [1,1.3,2.7,3,6,6,19,1,60,140,140]
coeffarray = [1,1.1,1.9,2,3,3,7.5,1,17,38,40]

#xexpo1slider = plt.axes([0.25,0.05,0.65,0.02])
#xexpo2slider = plt.axes([0.25,0.07,0.65,0.02])
#xexpo3slider = plt.axes([0.25,0.09,0.65,0.02])
#xexpo4slider = plt.axes([0.25,0.11,0.65,0.02])
#xexpo5slider = plt.axes([0.25,0.13,0.65,0.02])
#xexpo6slider = plt.axes([0.25,0.15,0.65,0.02])
#xexpo7slider = plt.axes([0.25,0.17,0.65,0.02])
#xexpo8slider = plt.axes([0.25,0.19,0.65,0.02])
#xexpo9slider = plt.axes([0.25,0.21,0.65,0.02])
#xexpo10slider = plt.axes([0.25,0.23,0.65,0.02])
#xexpo11slider = plt.axes([0.25,0.25,0.65,0.02])
#expo1slider = Slider(xexpo1slider,'expo1',1/2,1,valinit=1)
#expo2slider = Slider(xexpo2slider,'expo2',1/3,1,valinit=1)
#expo3slider = Slider(xexpo3slider,'expo3',1/3,1,valinit=1)
#expo4slider = Slider(xexpo4slider,'expo4',1/4,1,valinit=1)
#expo5slider = Slider(xexpo5slider,'expo5',1/10,1,valinit=1)
#expo6slider = Slider(xexpo6slider,'expo6',1/10,1,valinit=1)
#expo7slider = Slider(xexpo7slider,'expo7',1/20,1,valinit=1)
#expo8slider = Slider(xexpo8slider,'expo8',1/50,1,valinit=1)
#expo9slider = Slider(xexpo9slider,'expo9',1/60,1,valinit=1)
#expo10slider = Slider(xexpo10slider,'expo10',1/150,1,valinit=1)
#expo11slider = Slider(xexpo11slider,'expo11',1/150,1,valinit=1)


plotfunction([1,0.86,0.57,0.51,0.34,0.33,0.13,1,0.06,0.03,0.02])
expolist = [1,1,1,1,1,1,1,1,1,1,1]
#def update(val):
#    expolist[0] = expo1slider.val
#    expolist[1] = expo2slider.val
#    expolist[2] = expo3slider.val
#    expolist[3] = expo4slider.val
#    expolist[4] = expo5slider.val
#    expolist[5] = expo6slider.val
#    expolist[6] = expo7slider.val
#    expolist[7] = expo8slider.val
#    expolist[8] = expo9slider.val
#    expolist[9] = expo10slider.val
#    expolist[10] = expo11slider.val
#    ax.clear()
#    plotfunction(expolist)
#    #fig.canvas.draw_idle()
#expo1slider.on_changed(update)
#expo2slider.on_changed(update)
#expo3slider.on_changed(update)
#expo4slider.on_changed(update)
#expo5slider.on_changed(update)
#expo6slider.on_changed(update)
#expo7slider.on_changed(update)
#expo8slider.on_changed(update)
#expo9slider.on_changed(update)
#expo10slider.on_changed(update)
#expo11slider.on_changed(update)
plt.tick_params(labelsize=18,top=True,right=True)
ax.set_xlim(0,960)
ax.set_xlabel(r'$U-U_{max} (mm/s)$',fontsize=24,labelpad=0)
ax.set_ylabel(r'$L/L_{\eta_{out}}$',fontsize=24,labelpad=0)

#ax.scatter(np.log(viscosityset), np.log(coeffarray),facecolor='none',edgecolor='C0')
plt.show() 

