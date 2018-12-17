from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from error_boxes import make_error_boxes as meb
from scipy import optimize
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
data261= np.genfromtxt('../2018_2_21_26cP_3colors/data1.csv',delimiter=',',names=True)
data262= np.genfromtxt('../2018_2_21_26cP_3colors/data2.csv',delimiter=',',names=True)
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
dataset = [data572,data512,data285,data214,data162,data92,data68,data65,data31,data26]
rhoset = [rho572,rho512,rho285,rho214,rho162,rho92,rho68,rho65,rho31,rho26]
gammaset = [gamma572,gamma512,gamma285,gamma214,gamma162,gamma92,gamma68,gamma65,gamma31,gamma26]
viscosityset = [572,512,285,214,162,92,68,65,31,26]

drg = zip(dataset,rhoset,gammaset,viscosityset)
fig,ax = plt.subplots(figsize=(2,1.5))
plt.subplots_adjust(bottom=0.3,left=0.3)

correction = 6.93-2.2
correction2= 18.59-4.73

cmap = plt.get_cmap('tab20')
umaxlist = []
umaxerrlist = []
mulist = []
def plotfunction(expo1,expo2,c):
    for i,(data,rho,gamma,mu) in enumerate(drg[:]):
        #if i not in [0,2,5,8]:
        #    continue
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
            h2max += correction
            h3max += correction
            pass
        u = np.copy(data['tapevelocity'])
        du = np.copy(data['velocityuncertainty'])
        dphi=(0.5*np.pi/180)*du/du
        etha = np.repeat(mu,len(u)) 

        #ignore np.nan values in the arrays
        u = u[~np.isnan(phi)]
        if len(u) == 0:
            continue
        du = du[~np.isnan(phi)]
        dphi = dphi[~np.isnan(phi)]
        phi = phi[~np.isnan(phi)]
            
        #xerr=np.vstack((du,du))
        #yerr=(dphi*np.sin(phi)/(np.cos(phi)**2*np.power(mu,expo1)))
        #yerr = np.vstack((yerr,yerr))
        #meb(ax,u,1/(np.cos(phi)*np.power(mu,expo1)),xerror=xerr,yerror=yerr,facecolor=cmap(i),errorcolor='None',alpha=1)
        #ax.errorbar(u,1/(np.cos(phi)*np.power(mu,expo1)),xerr=du,yerr=dphi*np.sin(phi)/(np.cos(phi)**2*np.power(mu,expo1)),fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i),ecolor=cmap(i),alpha=1,label='%scP'%mu,zorder=0)
        #ax.set_xlim(0,950)
        #ax.set_ylim(0,4.5)
        #ax.legend(markerscale=1)

        U = np.arange(0,np.max(u)*1e-3,0.001)
        etha = 18.27e-6 
        Ca = etha*U/gamma
        #y=a*x^b
        #logy = loga+b*logx
        fitfunc = lambda p, x: p*x
        errfunc = lambda p, x, y,err: (y-fitfunc(p,x))/err
        pinit = mu**0.75
        out = optimize.leastsq(errfunc, pinit, args = (u,1/np.cos(phi),du),full_output=1)
        pfinal = out[0]
        umax = 1/pfinal[0]
        cov_x=out[1]
        infodict = out[2]
        s_sq = (infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(pfinal))
        covar = cov_x*s_sq
        perr = np.sqrt(covar[0][0])
        umaxerr = perr/(pfinal[0]**2)
        print 'umax = %.2f +- %.2f'%(umax,umaxerr)
        umaxlist.append(umax)
        umaxerrlist.append(umaxerr)
        mulist.append(mu)

        #ax.errorbar(mu,umax,xerr=0,yerr=umaxerr,fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i),ecolor=cmap(i),alpha=1,label='%scP'%mu,zorder=0)
        ax.errorbar(np.log(mu),np.log(umax),xerr=0,yerr=umaxerr/(umax),fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i),ecolor=cmap(i),alpha=1,label='%scP'%mu,zorder=0)
        #ax.plot(1e3*U,pfinal*1e3*U,color=cmap(i/20))
        #print 'b = %.2f+/-%.2f' %(b,berr),'for mu',mu
#xexpo1slider = plt.axes([0.25,0.08,0.65,0.03])
#xexpo2slider = plt.axes([0.25,0.05,0.65,0.03])
#xcslider = plt.axes([0.25,0.02,0.65,0.03])
#expo1slider = Slider(xexpo1slider,'expo1',0,1,valinit=3/4)
#expo2slider = Slider(xexpo2slider,'expo2',0,1,valinit=1/3)
#cslider = Slider(xcslider,'c',0,30,valinit=23)
plotfunction(0,1/3,23)
umax = np.array(umaxlist)
umaxerr = np.array(umaxerrlist)
muarray =np.array(mulist) 
logmu = np.log(muarray)
dmu=0*muarray
logmuerr = dmu/muarray
logumax = np.log(umax)
logumaxerr = umaxerr/umax
#Umax = c*mu^d
#logUmax = logc+d*logmu
fitfunc = lambda p,x: p[0]+p[1]*x
errfunc = lambda p,x,y,err: (y-fitfunc(p,x))/err
init = np.exp(9),-0.75
out = optimize.leastsq(errfunc,init,args=(logmu,logumax,logumaxerr),full_output=1)
pfinal = out[0]
cov_x = out[1]
c = np.exp(pfinal[0])
d = pfinal[1]
infodict = out[2]
s_sq = (infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(pfinal))
covar = cov_x*s_sq
cerr = np.sqrt(covar[0][0])*c
logcerr = cerr/c
derr = np.sqrt(covar[1][1])
plt.plot(logmu,np.log(c)+(d)*logmu)
print 'logc = %.4f+/-%.4f' %(np.log(c),logcerr)
print 'd = %.4f+/-%.4f' %(d,derr)

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

plt.tick_params(labelsize=12)
plt.axis('equal')

#ax.set_xlabel('$U (mm/s)$')
#ax.set_ylabel(r'$\frac{1}{\cos(\phi)}$',fontsize=15)
plt.show() 

