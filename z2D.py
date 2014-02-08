####### Jia Liu 2014/2/5 ########################
###### convert between z and comoving distances #

from scipy import *
from scipy import optimize
from scipy.integrate import quad

######## begin parameters ########
c = 299792.458 #speed of light in km/s
H0=72.0 #hubble constant
OmegaM=0.26
OmegaV=1-OmegaM #assume flat universe
######## end parameters ########

######## begin functions ######
H_inv = lambda z: 1.0/(H0*sqrt(OmegaM*(1+z)**3+OmegaV)) # calculate 1/H(z)
#from z to luminosity distance DL, or comoving distance Dc in Mpc
DL = lambda z: (1+z)*c*quad(H_inv, 0, z)[0]# luminosity distance in Mpc
Dc= lambda z: c*quad(H_inv, 0, z)[0]# comoving dictance in Mpc
#from Dc in Mpc to z
Dcroot = lambda z, Dc0: Dc(z)-Dc0# used by Dc2z to find root
Dc2z = lambda Dc0: (optimize.root(Dcroot,0.1,args=(Dc0),method='hybr')).x # find z for a certain comoving distance Dc0 (in Mpc)
######## end functions ######

### actual calculations ######
#### our lensing planes comoving distance: first 40, then one per 80 Mpc
Dc_arr = arange(40,5350,80)# these are our lensing planes
z_arr = array(map(Dc2z,Dc_arr))# these are corresponding redshifts for Dc in Dc_arr
print (array([Dc_arr,z_arr]).T)

##### test validity (pass)
##### amax(abs(map(Dc,z_arr)-Dc_arr))
