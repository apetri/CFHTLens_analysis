import sys,os
sys.modules["mpi4py"] = None
from lenstools.simulations import PotentialPlane
import numpy as np
import astropy.units as u
#from astropy.cosmology import WMAP9
from lenstools.pipeline.simulation import LensToolsCosmology
from scipy.integrate import quad
from scipy import *

info_fn = lambda cosmo: '/scratch/02977/jialiu/CMB_hopper/CMB_batch_storage/%s/1024b600/ic1/Planes/info.txt'%(cosmo)

#Size of the plane (make it large enough to fit 3.5 degrees in angular size)
iangle = 1000*u.Mpc

cosmo_arr = genfromtxt('/scratch/02977/jialiu/CMB_hopper/CMB_batch/cosmo_list.txt',dtype='string')
h = 0.72
c = 299792.458#km/s
for cosmo in cosmo_arr[4:]:
    print cosmo
    Om = float(cosmo[2:7])
    w = float(cosmo[17:23])
    si8 = float(cosmo[-5:])
    
    if Om==0.296:
        continue
    #Om, w, si8=[0.296,-1.000,0.786]
    cosmology = LensToolsCosmology(Om0=Om,Ode0=1-Om,w0=w,sigma8=si8)
    
    H_inv = lambda z: 1.0/(72.0*sqrt(Om*(1+z)**3+(1-Om)))
    DC_fcn = lambda z: c*quad(H_inv, 0, z)[0]
    
    #Write 9 fake planes with zeroes on them (if the last non trivial snapshot is 60, these will be labelled by 61)
    last_snapshot=len(genfromtxt(info_fn(cosmo), dtype='string'))
    
    for normal in [0,1,2]:
        for cut_point in [0,1,2]:
            print normal, cut_point
            #pln = PotentialPlane(data=np.zeros((4096,4096)),angle=iangle,redshift=1101.0,cosmology=cosmology,num_particles=1024**3/3.)
            #pln.save('snap{0}_potentialPlane{1}_normal{2}.fits'.format(last_snapshot+1,cut_point,normal))
            os.system("ln -sf /home1/02977/jialiu/scratch/CMB_hopper/CMB_batch_storage/Om0.296_Ol0.704_w-1.000_si0.786/1024b600/ic1/Planes/snap58_potentialPlane{0}_normal{1}.fits /home1/02977/jialiu/scratch/CMB_hopper/CMB_batch_storage/{2}/1024b600/ic1/Planes/snap{3}_potentialPlane{0}_normal{1}.fits".format(cut_point, normal, cosmo, last_snapshot))
    
    string = "s=%i,d=%.8f Mpc/h,z=1101.0\n"%(last_snapshot, DC(1101)*h)
    with open(info_fn(cosmo), "a") as myfile: myfile.write(string)


def gen_infotxt(cosmo):
    '''for messed up info.txt in scratch, using gadget scales to generate'''
    a_gadget=genfromtxt('/home1/02977/jialiu/work/CMB_hopper/CMB_batch/%s/1024b600/ic1/outputs.txt'%(cosmo))
    z_arr=1/a_gadget-1
    Om = float(cosmo[2:7])
    w = float(cosmo[17:23])
    si8 = float(cosmo[-5:])
    H_inv = lambda z: 1.0/(72.0*sqrt(Om*(1+z)**3+(1-Om)))
    DC_fcn = lambda z: c*quad(H_inv, 0, z)[0]
    info_file = info_fn(cosmo) 
    f=open(info_file,'w')
    i=0
    for z in z_arr:
        string = "s=%i,d=%.8f Mpc/h,z=%.8f\n"%(i, DC_fcn(z)*h,z)
        f.write(string)
        i+=1
    f.close()
#Once you do this, open the file 'info.txt', and add a line
#s=61,d=<the_comoving_distance in Mpc/h> Mpc/h,z=<the_redshift>
#It doesn't matter that s=61 has a higher redshift than s=60, since the raytracing orders the lenses before starting the calculations