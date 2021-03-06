from __future__ import division
from scipy import optimize
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from leastsq import leastsq_weighted
from leastsq import chi2test 
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
params= {'text.latex.preamble' : [r'\usepackage{amsmath}']}
plt.rcParams.update(params)
rc('text',usetex=True)
rc('font',family='serif')
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
correction = 6.93-2.2
correction2= 18.59-4.73

drg = zip(dataset,rhoset,gammaset,viscosityset)
"""
for data inspection

"""
#fig,axes = plt.subplots(2,2,figsize=(12,8))
#ax,ax2,ax3,ax4 = axes.flat[:]
"""
for data plotting

"""
fig,ax = plt.subplots(figsize=(11.55,6))

plt.subplots_adjust(bottom=0.2,left=0.2)
#plt.subplots_adjust(bottom=0.3,top=0.8,left=0.35)
cmap = plt.get_cmap('tab20')
#plt.xlim(20,1000)
#plt.ylim(0.1,160)
alist = []
aerrlist = []
blist = []
berrlist = []
mulist = []
markerlist = ['o','>','^','<','v','s','H','*','p','D','h']
resultarray = np.load('umaxresult.npy')
muarray = resultarray[0,:]
umaxarray= resultarray[1,:]
umaxerrarray= resultarray[2,:]

for i,(data,rho,gamma,mu) in enumerate(drg[:]): #x[1:2] is [x[1]], an array
    phi = ((np.pi/180)*(data['leftangle']+data['rightangle'])/2)
    height = (data['lefth']+data['righth'])/2
    hflat = (data['rightflat']+data['leftflat'])/2
    hmin = (data['rightmin']+data['leftmin'])/2
    u = np.copy(data['tapevelocity'])
    
    du = np.copy(data['velocityuncertainty'])
    etha = np.repeat(mu,len(u)) 

    #if mu == 31:
    #    phi31= ((np.pi/180)*(data['leftangle']+data['rightangle'])/2)
    #    height31= (data['lefth']+data['righth'])/2
    #    hflat31= (data['rightflat']+data['leftflat'])/2
    #    hmin31= (data['rightmin']+data['leftmin'])/2
    #    u31= np.copy(data['tapevelocity'])
    #    du31= np.copy(data['velocityuncertainty'])
    #    hflat31 += correction
    #    hmin31  += correction
    #    continue
    #if mu == 26:
    #    phi = np.concatenate((phi,phi31))
    #    height= np.concatenate((height,height31))
    #    hflat= np.concatenate((hflat,hflat31))
    #    hmin= np.concatenate((hmin,hmin31))
    #    u= np.concatenate((u,u31))
    #    du= np.concatenate((du,du31))

    if mu == 31:
        hflat += correction
        hmin  += correction
    if mu == 153:
        hflat -= correction2
        hmin  -= correction2
    if mu == 68: # too few points for 68
        continue

    #ignore np.nan values in the arrays
    u = u[~np.isnan(hflat)]
    if len(u) == 0:
        continue
    du = du[~np.isnan(hflat)]
    hflat = hflat[~np.isnan(hflat)]
    #for s in range(0):
    #    maxindex = u.argmax()
    #    u = np.delete(u,maxindex)
    #    hflat = np.delete(hflat, maxindex)
    #    du = np.delete(du, maxindex)
    #for s in range(0):
    #    minindex = u.argmin()
    #    u = np.delete(u,minindex)
    #    hflat = np.delete(hflat, minindex)
    #    du = np.delete(du, minindex)

    U = np.arange(np.min(u)*1e-3,np.max(u)*1e-3,0.001)
    etha_air = 18.27e-6 
    Ca = etha_air*U/gamma
    #h=a*u^b
    #logh = loga+b*logu
    log10uerr = du/(u*np.log(10))
    #log10herr = (0.59/2)/(hflat*np.log(10))
    viscosityset = [572,512,285,214,162,153,92,68,65,31,26]
    #n = np.array([0.5,0.5,0.5,1,1,1,1.5,3,3,3,3])/2
    fluc = np.sqrt(0.5**2+np.power(np.array([0.4, 0.4, 0.7, 0.9, 1.1, 1.2, 1.8, 2.3, 2.4, 4.4, 5.1])/2,2)) #0.5 is error for identifying the wrong stripe
    #fluc = [1,1,1,1,1,1,1,1,1,1,1]
    log10herr = (fluc[i]*0.53/2)/(hflat*np.log(10))
    log10a, b,sigmalog10a, sigmab  = leastsq_weighted(np.log10(1e-3*u),np.log10(1e-6*hflat),log10uerr,log10herr)
    a = 10**(log10a)
    berr = sigmab
    sigmaa = sigmalog10a*a*np.log(10)
    aerr = sigmaa
    alist.append(a)
    blist.append(b)
    aerrlist.append(sigmaa)
    berrlist.append(sigmab)
    mulist.append(mu)
    prob = chi2test(np.log10(1e-3*u),np.log10(1e-6*hflat),log10uerr,log10herr)
    #print 'b = %.2f+/-%.2f' %(b,berr),'for mu',mu,'cP'

    #print 'chi2test:', prob*100
    print 'reduced chi2:', prob, 'for mu', mu, 'cP'
    #print 'a = %.6f+/-%.6f' %(a,aerr),'for mu',mu,'cP'

    ax.errorbar(np.log10(1e-3*(u)),np.log10(1e-6*hflat),fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor='C0',xerr=log10uerr, yerr=log10herr,capsize=4,label='%scP'%mu,zorder=1)
    ax.plot(np.log10(1e-3*u),np.log10(a)+(b)*np.log10(1e-3*u),color=cmap(i),zorder=0)
    #ax.plot(np.log10(U), np.log10(0.93*np.sqrt(gamma/(rho*9.8))*np.power(Ca, 2/3)),color=cmap(i),zorder=0,ls=(0,(3,5)))
    #ax.legend(ncol=2)
    #ax2.errorbar(u,hflat,fmt=markerlist[i],ms=8,markeredgewidth=1,mfc=cmap(i/float(20)),ecolor='C0',xerr=du,yerr=0.59/2*n[i],capsize=4,label='%scP'%mu)
    #ax3.errorbar(mu,b,fmt=markerlist[i],ms=8,markeredgewidth=1,mfc=cmap(i),ecolor='C0',xerr=0,yerr=berr,capsize=4,label='%scP'%mu)
    #ax3.set_ylim(0,0.9)
    #ax4.errorbar(np.log10(mu),np.log10(a),fmt=markerlist[i],ms=8,mew=1,mfc=cmap(i),ecolor='C0',xerr=0, yerr=aerr/(a*np.log(10)),capsize=4,label='%scP'%mu)
    #ax4.set_xlabel(r'$log(\eta_{liq})$',fontsize=18)
    #ax4.set_ylabel(r'$log(A)$',fontsize=18)

aarray = np.array(alist)
aerrarray = np.array(aerrlist)
barray = np.array(blist)
muarray =np.array(mulist) 
bmean = np.mean(barray)
berrfinal = np.sqrt(np.sum([x**2 for x in berrlist]))/len(berrlist)
#berrfinal = np.std(barray)
print 'b average is',bmean
print 'b standard error is',berrfinal 
log10mu = np.log10(muarray)
dmu=0.*muarray
log10muerr = dmu/(muarray*np.log(10))
log10a= np.log10(aarray)
log10aerr= aerr/(aarray*np.log(10))
#a = e*mu^f
#loga = loge +f*logmu
log10e, f, sigmalog10e, sigmaf = leastsq_weighted(log10mu,log10a,log10muerr,log10aerr)
#print 'loge = %.4f+/-%.4f' %(loge,sigmaloge)
print 'f = %.4f+/-%.4f' %(f,sigmaf)

#ax4.plot(log10mu, log10e+f*log10mu,'C0')
#ax3.axhline(y = bmean)
#ax3.axhline(y = 0.667,ls='--')
#ax3.axhline(y = 0.5,ls='--')
#ax3.set_xlabel(r'$\eta_{liq}$',fontsize=18)
#ax3.set_ylabel(r'$\alpha$',fontsize=18)
#ax3.axhspan(bmean-berrfinal,bmean+berrfinal,alpha =0.5)

plt.tick_params(labelsize=20,right=True,top=True)
ax.xaxis.set_label_coords(0.5,-0.1)
ax.yaxis.set_label_coords(-0.08,0.5)
ax.set_xlabel(r'$\log(U)$',fontsize=26,labelpad=0)
ax.set_ylabel(r'$\log(H_{\text{thin}})$',fontsize=26,labelpad=0)
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.set_xlim(-1.4,0.2)
#ax.set_ylim(-6,-4.7)
plt.show()

