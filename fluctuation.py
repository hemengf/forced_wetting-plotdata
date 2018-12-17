import numpy as np
import matplotlib.pyplot as plt
from leastsq import leastsq_unweighted as ls
flucdic = {}
vis = [34,41,53,63,75,95,298,340,539]
fluc1 = [3,4,2.7,1.7,1.2]
flucdic[34] = np.array([3,4,5])
flucdic[41] = np.array([2,4,5])
flucdic[53] = np.array([4,5,3])
flucdic[63] = np.array([3,2,3])
flucdic[75] = np.array([1,2,2])
flucdic[95] = np.array([1,2,0.5])
flucdic[298] = np.array([0.5,0.5,1])
flucdic[340] = np.array([0.5,0.5,1])
flucdic[539] = np.array([0.25,0.5,0.5])

A,B,_,_,_ = ls(np.log(vis),np.log([np.mean(flucdic[v]) for v in vis]))
newlogx = np.arange(min(np.log(vis)),max(np.log(vis)))
#plt.plot(np.log(vis),np.log([np.mean(flucdic[v]) for v in vis]),'o')
#plt.plot(newlogx, A+B*newlogx)

newx = np.arange(0,1000)
plt.plot(vis,[np.mean(flucdic[v]) for v in vis],'o')
plt.plot(newx, np.exp(A)*np.power(newx,B))
plt.ylim(0,6)

#calculate fluctuation:
for vis in [572,512,285,214,162,153,92,68,65,31,26]:
    print 'fluctuation=','{:.1f}'.format(np.exp(A)*np.power(vis,B)),'stripes','{:.1f}'.format(np.exp(A)*np.power(vis,B)*0.532/4),'um'

plt.show()
