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
gammaset = [gamma572,gamma512,gamma285,gamma214,gamma162,gamma153,gamma92,gamma68,gamma65,gamma31,gamma26]
viscosityset = [572,512,285,214,162,153,92,68,65,31,26]
drg = zip(dataset,rhoset,gammaset,viscosityset)
correction = 6.93-2.2
correction2= 18.59-4.73
correction3 = 14.87-6.65 #47-#21 experiments from 4751-4831
correction4 = 13.86-5.04 #44-#16 from 4831 on
correction5 = 14.87-11.7 #47-#37 for 4793only

'''
rmin = np.copy(data261['rightmin'])
lmin = np.copy(data261['leftmin'])
rflat = np.copy(data261['rightflat'])
lflat = np.copy(data261['leftflat'])
u = np.copy(data261['tapevelocity'])
du = np.copy(data261['velocityuncertainty'])
plt.errorbar(u,(rmin+lmin)/2,fmt='o',xerr=du,yerr =0.59/2)
plt.errorbar(u,(rflat+lflat)/2,fmt='o',mfc='none',xerr=du,yerr = 0.59/2)
#rmin-=correction3
#lmin-=correction3
#rflat-=correction3
#lflat-=correction3
#plt.errorbar(u,(rmin+lmin)/2,fmt='o',xerr=du,yerr =0.59/2)
#plt.errorbar(u,(rflat+lflat)/2,fmt='o',mfc='none',xerr=du,yerr = 0.59/2)

rmin = np.copy(data262['rightmin'])
lmin = np.copy(data262['leftmin'])
rflat = np.copy(data262['rightflat'])
lflat = np.copy(data262['leftflat'])
u = np.copy(data262['tapevelocity'])
du = np.copy(data262['velocityuncertainty'])
plt.errorbar(u,(rmin+lmin)/2,fmt='o',xerr=du,yerr =0.59/2)
plt.errorbar(u,(rflat+lflat)/2,fmt='o',mfc='none',xerr=du,yerr = 0.59/2)
#rmin-=correction4
#lmin-=correction4
#rflat-=correction4
#lflat-=correction4
#plt.errorbar(u,(rmin+lmin)/2,fmt='o',xerr=du,yerr =0.59/2)
#plt.errorbar(u,(rflat+lflat)/2,fmt='o',mfc='none',xerr=du,yerr = 0.59/2)
'''

fig,ax = plt.subplots(figsize=(5,5))
plt.subplots_adjust(bottom=0.2,left=0.2)
cmap = plt.get_cmap('tab20')
markerlist = ['o','>','^','<','v','s','H','*','p','D','h']
resultarray = np.load('umaxresult.npy')
muarray = resultarray[0,:]
umaxarray= resultarray[1,:]
umaxerrarray= resultarray[2,:]
def plotfunction(expo1,expo2,c):
    for i,(data,rho,gamma,mu) in enumerate(drg[:]):
        plt.sca(ax)
        phi = ((np.pi/180)*(data['leftangle']+data['rightangle'])/2)
        height = (data['lefth']+data['righth'])/2
        hflat = (data['rightflat']+data['leftflat'])/2
        hmin = (data['rightmin']+data['leftmin'])/2
        h1max = np.copy(data['1stmax'])
        h2max = np.copy(data['2ndmax'])
        h3max = np.copy(data['3rdmax'])


        if mu == 68:
            continue
        if mu == 31:
            hflat += correction
            hmin += correction
            pass
        if mu == 153:
            hflat -= correction2
            hmin -= correction2
            pass
        u = np.copy(data['tapevelocity'])
        du = np.copy(data['velocityuncertainty'])
        etha = np.repeat(mu,len(u)) 

        u = u[~np.isnan(hflat)]
        if len(u) == 0: 
            continue
        du = du[~np.isnan(hflat)]
        hflat = hflat[~np.isnan(hflat)]
            
        fluc = np.sqrt(0.5**2+np.power(np.array([0.4, 0.4, 0.7, 0.9, 1.1, 1.1, 1.8, 2.3, 2.4, 4.4, 5.2])/2,2))
        #fluc = np.sqrt(0.5**2+np.power(np.array([1,1,1,1,1,1,1,1,1,1,1])/2,2))
        #meb(ax,u,hflat,xerror=np.vstack((du,du)),yerror=(0.59/2)*np.ones((2,len(du))),facecolor=cmap(i+a),errorcolor='None',alpha=1)
        ax.errorbar(u,hflat,fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor='C0',xerr=du, yerr=fluc[i]*0.59/2,capsize=4,label='%scP'%mu,zorder=0)#for legend
        #ax.scatter(u,h1max,marker='o',s=100,facecolors='none',edgecolors='#1f77b4')#for better visualization 
        #ax.errorbar(u,hflat,fmt='s',ms=8,mfc=cmap(i+a),ecolor=cmap(i+a),xerr=du, yerr=0.59/2,alpha=1,label='%scP'%mu)
        #ax.errorbar(u,hmin,fmt='v',mfc='none',mec=cmap(i/float(20)),ecolor=cmap(i/float(20)),xerr=du,yerr=0.59/2,alpha=1)
        #ax.errorbar(u,h1max,fmt='v',ms=8,mfc=cmap(i/float(20)),ecolor=cmap(i/float(20)),xerr=du,yerr=0.59/2,alpha=1)
        ax.legend()

        U = np.arange(np.min(u)*1e-3,np.max(u)*1e-3,0.001)
        etha = 18.27e-6 
        Ca = etha*U/gamma
        plt.plot(1e3*U, 1e6*0.94*np.sqrt(gamma/(rho*9.8))*np.power(Ca, 2/3),color=cmap(i),zorder=2,lw=1,linestyle='--')
        #plt.plot(1e3*U,c*1e6*np.sqrt(gamma/(rho*9.8))*np.power(Ca, expo1)*(etha/(mu*1e-3))**(expo2),color=cmap(i),linestyle='-',linewidth=1,zorder=3)


#xexpo1slider = plt.axes([0.25,0.08,0.65,0.03])
#xexpo2slider = plt.axes([0.25,0.05,0.65,0.03])
#xcslider = plt.axes([0.25,0.02,0.65,0.03])
#expo1slider = Slider(xexpo1slider,'expo1',0,1,valinit=2/3)
#expo2slider = Slider(xexpo2slider,'expo2',0,1,valinit=1/3)
#cslider = Slider(xcslider,'c',0,30,valinit=23)
#
#
plotfunction(2/3,1/3,23)
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
plt.tick_params(labelsize=18)

#ax.set_xlim(0,950)
#ax.set_ylim(0,180)
ax.set_xlabel('$U (mm/s)$',fontsize=18)
ax.set_ylabel('$H_{thin}(\mu m)$',fontsize=18)
ax.set_xticks([0,250,500,750,1000])
plt.show()


