import numpy as np
import matplotlib.pyplot as plt
from leastsq import leastsq_weighted
fig,ax = plt.subplots(figsize=(3.0,3.5))
plt.subplots_adjust(bottom=0.4,left=0.5)
"""
    for 572cSt, the factor is 1, skip;
"""
factor_array = np.array([0.86,0.57,0.51,0.34,0.33,0.13,0.06,0.03,0.02])
factor_err_top = np.array([0.94,0.63,0.55,0.38,0.38,0.14,0.07,0.04,0.03])
factor_err_bot = np.array([0.78,0.54,0.49,0.32,0.32,0.12,0.05,0.02,0.01])
errplus = factor_err_top-factor_array
errminus= factor_array-factor_err_bot
err = np.vstack((errplus, errminus))
log10err = err/(factor_array*np.log(10))
viscosityset = np.array([512,285,214,162,153,92,65,31,26])


#factor=a*eta^b
#log(factor)=log(a)+b*log(eta)

log10a, b,sigmalog10a, sigmab  = leastsq_weighted(np.log10(viscosityset),np.log10(factor_array),0,2*log10err[1,:])
print b, log10a, sigmab

ax.errorbar(np.log10(viscosityset), np.log10(factor_array),yerr=log10err,fmt='o',ls='none',mfc='none',ms=8,mew=1,ecolor='C0',capsize=2)
ax.plot(np.log10(viscosityset), np.log10(np.power(viscosityset,1.26))-3.33)
ax.set_xticks([1.5,2.5])
ax.set_yticks([-2,0])
ax.yaxis.set_label_coords(-0.15,0.5)
#ax.annotate('1.25',xy=(100,0.02),fontsize=18)
plt.tick_params(labelsize=18,right=True,top=True)
ax.set_xlabel(r'$log(\eta_{out})$',fontsize=24,labelpad=0)
ax.set_ylabel(r'$log(L_{\eta_{out}})$',fontsize=24,labelpad=0)

plt.show()
