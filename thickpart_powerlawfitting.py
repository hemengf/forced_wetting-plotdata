from __future__ import division
from scipy import optimize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#data92 = np.genfromtxt('../2017_6_18_more/old_calibration/data.csv', delimiter=',',names=True,comments='#')
#data65 = np.genfromtxt('../2017_7_25_65cP/data.csv',delimiter=',',names=True,comments='#')
#data214 = np.genfromtxt('../2017_8_29_214cP/data.csv',delimiter=',',names=True,comments='#')
#data572 = np.genfromtxt('../2017_9_5_572cP/data.csv',delimiter=',',names=True,comments='#')
#data512 = np.genfromtxt('../2017_9_14_512cP/data.csv',delimiter=',',names=True,comments='#')
#data285 = np.genfromtxt('../2017_9_17_285cP/data.csv',delimiter=',',names=True,comments='#')
#data162 = np.genfromtxt('../2017_9_19_162cP/data.csv',delimiter=',',names=True,comments='#')
#data68 = np.genfromtxt('../2017_9_24_68cP/data.csv',delimiter=',',names=True,comments='#')
#data31 = np.genfromtxt('../2017_9_26_31cP/data.csv',delimiter=',',names=True,comments='#')
#data153 = np.genfromtxt('../2018_1_31_3colors_153cP/data.csv',delimiter=',',names=True,comments='#')
#data26= np.genfromtxt('../2018_2_21_26cP_3colors/data.csv',delimiter=',',names=True,comments='#')
#data261= np.genfromtxt('../2018_2_21_26cP_3colors/data1.csv',delimiter=',',names=True,comments='#')
#data262= np.genfromtxt('../2018_2_21_26cP_3colors/data2.csv',delimiter=',',names=True,comments='#')
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
dataset = [data572,data512,data285,data214,data162,data92,data68,data65,data31,data26]
rhoset = [rho572,rho512,rho285,rho214,rho162,rho92,rho68,rho65,rho31,rho26]
gammaset = [gamma572,gamma512,gamma285,gamma214,gamma162,gamma92,gamma68,gamma65,gamma31,gamma26]
viscosityset = [572,512,285,214,162,92,68,65,31,26]
correction = 6.93-2.2

drg = zip(dataset,rhoset,gammaset,viscosityset)
fig,ax = plt.subplots(figsize=(7.5,6.5))
plt.subplots_adjust(bottom=0.2)
cmap = plt.get_cmap('tab20')
#plt.xlim(20,1000)
#plt.ylim(0.1,160)

alist = []
aerrlist = []
blist = []
berrlist = []
mulist = []
for i,(data,rho,gamma,mu) in enumerate(drg[:]): #x[1:2] is [x[1]], an array
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
    if len(u) < 3 : 
        continue
    du = du[~np.isnan(h1max)]
    h1max= h1max[~np.isnan(h1max)]
    
    #ax.errorbar(u,h1max,fmt='s',ms=7,mew=0.5,mfc=cmap(i),ecolor=cmap(i),xerr=du, yerr=0.59/2,alpha=1,label='%scP'%mu,zorder=0)#for legend
    #ax.legend()
    U = np.arange(np.min(u)*1e-3,np.max(u)*1e-3,0.001)
    etha_air = 18.27e-6 
    Ca = etha_air*U/gamma
    #y=a*x^b
    #logy = loga+b*logx
    loguerr = du/u
    logherr = 0.59/(2*h1max)
    fitfunc = lambda p, x: p[0] + p[1]*x
    errfunc = lambda p, x, y,err: (y-fitfunc(p,x))/err
    pinit = [np.log(np.sqrt(3)*np.sqrt(gamma/(rho*9.8))*(etha_air/gamma)**(1/2)),1/2]
    out = optimize.leastsq(errfunc, pinit, args = (np.log(1e-3*u),np.log(1e-6*h1max),logherr),full_output=1)
    pfinal = out[0]
    covar=out[1]
    a = np.exp(pfinal[0])
    b = pfinal[1]
    aerr = np.sqrt(covar[0][0])*a
    berr = np.sqrt(covar[1][1])
    alist.append(a)
    blist.append(b)
    aerrlist.append(aerr)
    berrlist.append(berr)
    mulist.append(mu)
    print 'b = %.2f+/-%.2f' %(b,berr),'for mu',mu
    #ax.errorbar(mu,a,fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i/20),ecolor=cmap(i/20),xerr=0.*mu,yerr=aerr)
    ax.errorbar(np.log(mu),np.log(a),fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i),ecolor=cmap(i),xerr=0.,yerr=aerr/a)
    #ax.errorbar(mu,b,fmt='s',ms=7,markeredgewidth=0.5,mfc=cmap(i/20),ecolor=cmap(i/20),xerr = 0.*mu,yerr = berr)


    #ax.errorbar(np.log(1e-3*u),np.log(1e-6*h1max),fmt='s',ms=7,mew=0.5,mfc=cmap(i),ecolor='None',xerr=loguerr, yerr=0.59/(2*h1max),alpha=1,label='%scP'%mu,zorder=0)#for legend
    #ax.plot(np.log(1e-3*u), np.log(a)+b*np.log(1e-3*u),color=cmap(i/float(20)))
    #plt.plot(1e3*U, 1e6*np.sqrt(3)*np.sqrt(gamma/(rho*9.8))*np.power(Ca, 1/2))

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
init = np.exp(-12),-0.25
out = optimize.leastsq(errfunc,init,args=(logmu,loga,logaerr),full_output=1)
pfinal = out[0]
covar = out[1]
e = np.exp(pfinal[0])
f = pfinal[1]
eerr = np.sqrt(covar[0][0])*e
logeerr = eerr/e
ferr = np.sqrt(covar[1][1])
print 'loge = %.4f+/-%.4f' %(np.log(e),logeerr)
print 'f = %.4f+/-%.4f' %(f,ferr)
ax.plot(logmu, np.log(e)+(f+ferr)*logmu,'C0')
ax.plot(logmu, np.log(e)+(f-ferr)*logmu,'C0')

#plt.axhline(y = bmean)
#plt.axhspan(bmean-berrfinal,bmean+berrfinal,alpha =0.5)

plt.show()


