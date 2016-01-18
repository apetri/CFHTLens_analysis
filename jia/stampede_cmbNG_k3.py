import WLanalysis
import numpy as np
from scipy import *
from emcee.utils import MPIPool

CMBlensing_dir = '/work/02977/jialiu/CMBnonGaussian/'

PPA = 2048.0/(sqrt(sizedeg)*60.0)
sigmaG = PPA*1.0
kmapGen = lambda r: WLanalysis.readFits('/scratch/02977/jialiu/CMB_hopper/CMB_batch_storage/Om0.296_Ol0.704_w-1.000_si0.786/1024b600/Maps/WLconv_z1100.00_%04dr.fits'%(r))

def k3Gen(r):
    print r
    kmap=kmapGen(r)
    kmap_smoothed=WLanalysis.smooth(kmap,sigmaG)
    k3=mean(kmap_smoothed**3)
    return k3

pool=MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

k3_arr = array(pool.map(k3Gen, range(1,10241)))
save(CMBlensing_dir+'k3_arr.npy',k3_arr)
pool.close()
print 'done-done-done'
