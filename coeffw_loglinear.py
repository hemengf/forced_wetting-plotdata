import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from leastsq import leastsq_weighted
fig,ax = plt.subplots(figsize=(3.0,3.5))
plt.subplots_adjust(bottom=0.4,left=0.5)
"""
    for 572cSt, the factor is 1, skip;
"""
factor_array = np.array([0.97,0.89,0.87,0.86,0.86,0.86,0.89,0.82,0.79])
factor_ave=np.mean(factor_array)
factor_logave=np.mean(np.log10(factor_array)) 
factor_err_top = np.array([0.99,0.91,0.88,0.87,0.88,0.88,0.90,0.84,0.81])
factor_err_bot = np.array([0.95,0.87,0.85,0.84,0.84,0.84,0.88,0.77,0.76])
errplus = factor_err_top-factor_array
errminus= factor_array-factor_err_bot
err = np.vstack((errplus, errminus))
log10err = err/(factor_array*np.log(10))
viscosityset = np.array([512,285,214,162,153,92,65,31,26])

#factor=a*eta^b
#log(factor)=log(a)+b*log(eta)

log10a, b,sigmalog10a, sigmab  = leastsq_weighted(np.log10(viscosityset),np.log10(factor_array),0,2*log10err[1,:])
print b, log10a,sigmab

ax.errorbar(viscosityset, np.log10(factor_array),yerr=log10err,fmt='o',ls='none',mfc='none',ms=8,mew=1,ecolor='C0',capsize=2)
#ax.plot(np.log10(viscosityset), np.log10(np.power(viscosityset,0.04))-0.14)
#ax.set_xticks([1.5,2.5])
ax.set_yticks([-0.3,0,0.3])
#ax.yaxis.set_label_coords(-0.1,0.5)
#ax.annotate('1.25',xy=(100,0.02),fontsize=18)
plt.tick_params(labelsize=18,right=True,top=True)
#ax.axhline(y=factor_logave)
ax.set_xlabel(r'$\eta_{out}$',fontsize=24,labelpad=0)
ax.set_ylabel(r'$log(W_{\eta_{out}})$',fontsize=24,labelpad=0)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

plt.show()
