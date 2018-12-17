import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
reducedchi2 = np.array([0.4876,0.1038,0.1197,0.1215,0.1237,0.3241,2.0339,1.4365,1.1240,0.7955])
#reducedchi2 = np.random.normal(size=1000)
plt.hist(reducedchi2,10,normed=True)

x = np.linspace(0,10,100)
df=2
plt.plot(x, chi2.pdf(x,df),'C1')
 

plt.show()
