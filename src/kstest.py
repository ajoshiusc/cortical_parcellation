from scipy import stats
import scipy as sp
import numpy as np
import os

save_dir = '/home/sgaurav/Documents/git_sandbox/cortical_parcellation/src'
rand_index=np.load(os.path.join(save_dir, 'rand_indexleft.npz'))['rand_index']
roilists=sp.load(os.path.join(save_dir, 'rand_indexleft.npz'))['roilists']
cnt=0
different=[]
slightly_different=[]
identical=[]

for r in range(rand_index.shape[0]):
    if r%2==0 and min(rand_index[r]) ==1.0:
        cnt+=1
    if  r %2==0 and min(rand_index[r]) !=1.0:
        mu=np.mean(rand_index[r])
        sigma=np.std(rand_index[r])
        rand_index[r]=(rand_index[r]-mu)/sigma
        rand_index[r]=np.arctanh(rand_index[r])
        rand_index[r][~np.isfinite(rand_index[r])] = 0
        mu=np.mean(rand_index[r+1])
        sigma=np.std(rand_index[r+1])
        rand_index[r+1]=(rand_index[r+1]-mu)/sigma
        rand_index[r+1]=np.arctanh(rand_index[r+1])
        rand_index[r+1][~np.isfinite(rand_index[r+1])] = 0
        D,p= stats.ranksums(rand_index[r],rand_index[r+1])
        if p< 0.05:
            different.append(roilists[cnt])
        elif p>=0.05 and p<=0.10:
            slightly_different.append(roilists[cnt])
        else:
            identical.append(roilists[cnt])
        cnt+=1

c=['differnt (p < 0.05)\n','slightly differnt (0.05 <= p <= 0.10)\n','identical (p > 0.10)\n']

with open("hypothesis_test.txt","w") as f:
    f.write(c[0])
    f.write('----------------------------------------------------------------------------------\n')
    for i in range(different.__len__()):
        f.write(different[i])
        f.write('\n')
    f.write('\n')
    f.write('\n')
    f.write(c[1])
    f.write('----------------------------------------------------------------------------------\n')
    for i in range(slightly_different.__len__()):
        f.write(slightly_different[i])
        f.write('\n')
    f.write('\n')
    f.write('\n')
    f.write(c[2])
    f.write('----------------------------------------------------------------------------------\n')
    for i in range(identical.__len__()):
        f.write(identical[i])
        f.write('\n')