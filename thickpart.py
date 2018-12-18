from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from error_boxes import make_error_boxes as meb
from matplotlib import rc
rc('text',usetex=True)
#data92 = np.genfromtxt('../2017_6_18_more/old_calibration/data.csv', delimiter=',',names=True)
#data65 = np.genfromtxt('../2017_7_25_65cP/data.csv',delimiter=',',names=True)
#data214 = np.genfromtxt('../2017_8_29_214cP/data.csv',delimiter=',',names=True)
#data572 = np.genfromtxt('../2017_9_5_572cP/data.csv',delimiter=',',names=True)
#data512 = np.genfromtxt('../2017_9_14_512cP/data.csv',delimiter=',',names=True)
#data285 = np.genfromtxt('../2017_9_17_285cP/data.csv',delimiter=',',names=True)
#data162 = np.genfromtxt('../2017_9_19_162cP/data.csv',delimiter=',',names=True)
#data68 = np.genfromtxt('../2017_9_24_68cP/data.csv',delimiter=',',names=True)
#data31 = np.genfromtxt('../2017_9_26_31cP/data.csv',delimiter=',',names=True)
#data153 = np.genfromtxt('../2018_1_31_3colors_153cP/data.csv',delimiter=',',names=True)
#data26= np.genfromtxt('../2018_2_21_26cP_3colors/data.csv',delimiter=',',names=True)
#data261= np.genfromtxt('../2018_2_21_26cP_3colors/data1.csv',delimiter=',',names=True)
#data262= np.genfromtxt('../2018_2_21_26cP_3colors/data2.csv',delimiter=',',names=True)
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
markerlist = ['o','>','^','<','v','s','H','*','p','D','h']
resultarray = np.load('umaxresult.npy')
muarray = resultarray[0,:]
umaxarray= resultarray[1,:]
umaxerrarray= resultarray[2,:]

correction = 6.93-2.2
correction2= 18.59-4.73

cmap = plt.get_cmap('tab20')
def plotfunction(expo1,expo2,c):
    for i,(data,rho,gamma,mu) in enumerate(drg[:]):
        a=0
        print mu
        plt.sca(ax)
        phi = ((np.pi/180)*(data['leftangle']+data['rightangle'])/2)
        height = (data['lefth']+data['righth'])/2
        hflat = (data['rightflat']+data['leftflat'])/2
        hmin = (data['rightmin']+data['leftmin'])/2
        h1max = np.copy(data['1stmax'])
        h2max = np.copy(data['2ndmax'])
        h3max = np.copy(data['3rdmax'])


        if mu == 31:
            hflat += correction
            hmin += correction
            h1max += correction
            pass
        u = np.copy(data['tapevelocity'])
        du = np.copy(data['velocityuncertainty'])
        etha = np.repeat(mu,len(u)) 

        u = u[~np.isnan(h1max)]
        if len(u) == 0: 
            continue
        du = du[~np.isnan(h1max)]
        h1max= h1max[~np.isnan(h1max)]
            
        #meb(ax,u,hflat,xerror=np.vstack((du,du)),yerror=(0.59/2)*np.ones((2,len(du))),facecolor=cmap(i+a),errorcolor='None',alpha=1)
        ax.errorbar(u,h1max,fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor='C0',xerr=du, yerr=0.59/2,alpha=1,label='%scP'%mu,zorder=0)#for legend
        #ax.scatter(u,h1max,marker='o',s=100,facecolors='none',edgecolors='#1f77b4')#for better visualization 
        #ax.errorbar(u,hflat,fmt='s',ms=8,mfc=cmap(i+a),ecolor=cmap(i+a),xerr=du, yerr=0.59/2,alpha=1,label='%scP'%mu)
        #ax.errorbar(u,hmin,fmt='v',mfc='none',mec=cmap(i/float(20)),ecolor=cmap(i/float(20)),xerr=du,yerr=0.59/2,alpha=1)
        #ax.errorbar(u,h1max,fmt='v',ms=8,mfc=cmap(i/float(20)),ecolor=cmap(i/float(20)),xerr=du,yerr=0.59/2,alpha=1)
        #ax.legend(loc=4)

        U = np.arange(np.min(u)*1e-3,np.max(u)*1e-3,0.001)
        etha = 18.27e-6 
        Ca = etha*U/gamma
        #plt.plot(1e3*U,c*1e6*np.sqrt(gamma/(rho*9.8))*np.power(Ca, expo1)*(etha/(mu*1e-3))**(expo2),color=cmap(i+a),linestyle='-',linewidth=1,zorder=3)


#xexpo1slider = plt.axes([0.25,0.08,0.65,0.03])
#xexpo2slider = plt.axes([0.25,0.05,0.65,0.03])
#xcslider = plt.axes([0.25,0.02,0.65,0.03])
#expo1slider = Slider(xexpo1slider,'expo1',0,1,valinit=2/3)
#expo2slider = Slider(xexpo2slider,'expo2',0,1,valinit=1/3)
#cslider = Slider(xcslider,'c',0,30,valinit=23)
#
#
plotfunction(1/2,-1/4,0.5)
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
plt.tick_params(labelsize=18,top=True,right=True)

ax.set_xlim(0,750)
ax.set_ylim(0,180)
ax.yaxis.set_label_coords(-0.15,0.5)
ax.set_xlabel(r'$U (mm/s)$',fontsize=24,labelpad=0)
ax.set_ylabel(r'$H_{max}(\mu m)$',fontsize=24,labelpad=0)
ax.set_yticks([0,50,100,150])
plt.show()


