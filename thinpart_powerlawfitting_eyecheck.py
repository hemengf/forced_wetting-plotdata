from __future__ import division
from scipy import optimize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from leastsq import leastsq_weighted
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
dataset = [data572,data512,data285,data214,data162,data153,data92,data68,data65,data31,data26]
rhoset = [rho572,rho512,rho285,rho214,rho162,rho153,rho92,rho68,rho65,rho31,rho26]
gammaset = [gamma572,gamma512,gamma285,gamma214,gamma162,gamma153,gamma92,gamma68,gamma65,gamma31,gamma26]
viscosityset = [572,512,285,214,162,153,92,68,65,31,26]
correction = 6.93-2.2
correction2= 18.59-4.73

drg = zip(dataset,rhoset,gammaset,viscosityset)
fig,axes = plt.subplots(2,2,figsize=(12,7.))
ax,ax2,ax3,ax4 = axes.flat[:]
#plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(bottom=0.03,top=0.97)
cmap = plt.get_cmap('tab20')
#plt.xlim(20,1000)
#plt.ylim(0.1,160)
alist = []
aerrlist = []
blist = []
berrlist = []
mulist = []
x1lo = [-3.34618,-2.66508,-2.28059,-2.07285,-1.85022,np.nan,-1.30903,np.nan,-1.13209,-0.641602,np.nan]
y1lo = [-13.5298,-13.3355,-13.0186,-12.8646,-12.488,np.nan,-12.0341,np.nan,-11.832,-11.1586,np.nan]
x2lo = [-2.00975,-1.84248,-1.52143,-1.45931,-1.04251,np.nan,-0.473412,np.nan,-0.475415,-0.105728,np.nan]
y2lo = [-13.0933,-13.0222,-12.785,-12.6178,-12.2018,np.nan,-11.6031,np.nan,-11.5545,-10.9365,np.nan]
x1hi= [-2.78287,-2.44645,-2.29567,-1.97536,-1.88369,np.nan,-1.37739,np.nan,-1.24067,-0.555253,np.nan]
y1hi= [-13.5262,-13.333,-13.0966,-12.8987,-12.5723,np.nan,-12.2658,np.nan,-12.0859,-11.2064,np.nan]
x2hi= [-2.01019,-1.84248,-1.52143,-1.45931,-1.04251,np.nan,-0.473412,np.nan,-0.475415,-0.109538,np.nan]
y2hi= [-13.0925,-13.0222,-12.785,-12.6178,-12.2018,np.nan,-11.6031,np.nan,-11.5545,-10.8237,np.nan]
for i,(data,rho,gamma,mu) in enumerate(drg[:-1]): #x[1:2] is [x[1]], an array
    phi = ((np.pi/180)*(data['leftangle']+data['rightangle'])/2)
    height = (data['lefth']+data['righth'])/2
    hflat = (data['rightflat']+data['leftflat'])/2
    hmin = (data['rightmin']+data['leftmin'])/2
    u = np.copy(data['tapevelocity'])
    du = np.copy(data['velocityuncertainty'])
    etha = np.repeat(mu,len(u)) 

    if mu == 31:
        hflat += correction
        hmin  += correction
    if mu == 153:
        hflat -= correction2
        hmin  -= correction2
        continue
    if mu == 68: # too few points for 68
        continue

    #ignore np.nan values in the arrays
    u = u[~np.isnan(hflat)]
    if len(u) == 0:
        continue
    du = du[~np.isnan(hflat)]
    hflat = hflat[~np.isnan(hflat)]
    for s in range(0):
        maxindex = u.argmax()
        u = np.delete(u,maxindex)
        hflat = np.delete(hflat, maxindex)
        du = np.delete(du, maxindex)
    for s in range(0):
        minindex = u.argmin()
        u = np.delete(u,minindex)
        hflat = np.delete(hflat, minindex)
        du = np.delete(du, minindex)

    #ax2.errorbar(u,hflat,fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i/float(20)),ecolor=cmap(i/float(20)),xerr=du,yerr = 0.59/2,label='%scP'%mu)
    U = np.arange(np.min(u)*1e-3,np.max(u)*1e-3,0.001)
    etha_air = 18.27e-6 
    Ca = etha_air*U/gamma
    #y=a*x^b
    #logy = loga+b*logx
    loguerr = du/u
    logherr = 0.59/(2*hflat)
    loga, b,sigmaloga, sigmab  = leastsq_weighted(np.log(1e-3*u),np.log(1e-6*hflat),loguerr,logherr)
    a = np.exp(loga)
    berr = sigmab
    sigmaa = sigmaloga*a
    aerr = sigmaa
    #fitfunc = lambda p, x: p[0] + p[1]*x
    #errfunc = lambda p, x, y,err: (y-fitfunc(p,x))/err
    #pinit = [np.log(0.94*np.sqrt(gamma/(rho*9.8))*(etha_air/gamma)**(2/3)),2/3]
    #out = optimize.leastsq(errfunc, pinit, args = (np.log(1e-3*u),np.log(1e-6*hflat),logherr),full_output=1)

    #'''
    #when fitting power law one has to be careful that y and x consistent in units. using inconsistent units will not affect b, but will affect a.  The effect depends on b, so when doing fits in a loop such as in this case, unless b's are exactly the same, a will be affected non-uniformly
    #'''
    #pfinal = out[0]
    #cov_x=out[1]
    #a = np.exp(pfinal[0])
    #b = pfinal[1]
    #infodict = out[2]
    #s_sq = (infodict['fvec']**2).sum()/(len(infodict['fvec'])-len(pfinal))
    #covar = cov_x*s_sq
    #aerr = np.sqrt(covar[0][0])*a
    #berr = np.sqrt(covar[1][1])
    alist.append(a)
    blist.append(b)
    aerrlist.append(aerr)
    berrlist.append(berr)
    mulist.append(mu)
    print 'b = %.2f+/-%.2f' %(b,berr),'for mu',mu,'cP'
    #print 'a = %.6f+/-%.6f' %(a,aerr),'for mu',mu,'cP'
    ax3.errorbar(mu,b,fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i),ecolor=cmap(i),xerr = 0.*mu,yerr = berr,label='%scP'%mu)
    ax3.set_ylim(0,0.9)
    #ax.errorbar(mu,a,fmt='s',ms=7,mfc=cmap(i/20),ecolor=cmap(i/20),xerr=0.*mu,yerr = aerr)
    #ax4.errorbar(np.log(mu),np.log(a),fmt='s',ms=7,mew=0.5,mfc=cmap(i),ecolor=cmap(i),xerr=0, yerr=aerr/a,alpha=1,label='%scP'%mu)

    ax.errorbar(np.log(1e-3*u),np.log(1e-6*hflat),fmt='s',ms=7,mew=0.5,mfc=cmap(i),ecolor=cmap(i),xerr=loguerr, yerr=logherr,alpha=1,label='%scP'%mu,zorder=0)
    ax2.errorbar(np.log(1e-3*u),np.log(1e-6*hflat),fmt='s',ms=7,mew=0.5,mfc=cmap(i),ecolor=cmap(i),xerr=loguerr, yerr=logherr,alpha=1,label='%scP'%mu,zorder=0)
    eyex = np.log(1e-3*u)
    ax2.plot(eyex,(eyex-x2lo[i])*(y2lo[i]-y1lo[i])/(x2lo[i]-x1lo[i])+y2lo[i],color=cmap(i))
    ax2.plot(eyex,(eyex-x2hi[i])*(y2hi[i]-y1hi[i])/(x2hi[i]-x1hi[i])+y2hi[i],color=cmap(i))
    berreyelo = b-(y2lo[i]-y1lo[i])/(x2lo[i]-x1lo[i])
    berreyehi = (y2hi[i]-y1hi[i])/(x2hi[i]-x1hi[i])-b
    ax4.errorbar(mu,b,fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i),ecolor=cmap(i),xerr = 0.*mu,yerr =np.array([[berreyehi],[berreyelo]]),label='%scP'%mu)
    ax4.set_ylim(0,0.9)
    #ax.plot(np.log(1e-3*u),np.log(a)+(b)*np.log(1e-3*u),color=cmap(i))
    ax.plot(np.log(1e-3*u),np.log(a+aerr)+(b+berr)*np.log(1e-3*u),color=cmap(i))
    ax.plot(np.log(1e-3*u),np.log(a-aerr)+(b-berr)*np.log(1e-3*u),color=cmap(i))


    #plt.loglog(1e3*U, 1e6*0.94*np.sqrt(gamma/(rho*9.8))*np.power(Ca, 2/3))
    #plt.plot(1e3*U, 1e6*np.sqrt(3)*np.sqrt(gamma/(rho*9.8))*np.power(Ca, 1/2))
    ax.legend()
#plt.plot(np.log(muforplot),-0.33*np.log(muforplot)-10,'C1-',)
aarray = np.array(alist)
aerrarray = np.array(aerrlist)
barray = np.array(blist)
muarray =np.array(mulist) 
bmean = np.mean(barray)
#berrfinal = np.sqrt(np.sum([x**2 for x in berrlist]))/len(berrlist)
berrfinal = np.std(barray)
print 'b average is',bmean
print 'b standard error is',berrfinal 
logmu = np.log(muarray)
dmu=0.*muarray
logmuerr = dmu/muarray
loga= np.log(aarray)
logaerr= aerr/aarray
#a = e*mu^f
#loga = loge +f*logmu
# y   = p[0] +p[1]x
fitfunc = lambda p,x: p[0]+p[1]*x
errfunc = lambda p,x,y,err: (y-fitfunc(p,x))/err
init = np.exp(-12),-0.5
out = optimize.leastsq(errfunc,init,args=(logmu,loga,logaerr),full_output=1)
pfinal = out[0]
cov_x = out[1]
e = np.exp(pfinal[0])
f = pfinal[1]
infodict = out[2]
s_sq = (infodict['fvec']**2).sum()/ (len(out[2]['fvec'])-len(out[0]))
covar = cov_x*s_sq
eerr = np.sqrt(covar[0][0])*e
logeerr = eerr/e
ferr = np.sqrt(covar[1][1])
#print 'loge = %.4f+/-%.4f' %(np.log(e),logeerr)
print 'f = %.4f+/-%.4f' %(f,ferr)
#ax.errorbar(logmu,loga,fmt='s',mfc='none',yerr = logaerr)
#ax4.plot(logmu, np.log(e)+f*logmu,'C0')

#ax3.axhline(y = bmean)
#ax3.axhspan(bmean-berrfinal,bmean+berrfinal,alpha =0.5)

plt.show()

