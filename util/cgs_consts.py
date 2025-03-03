import astropy.constants as const
import astropy.units as u
from astropy.cosmology import Planck13


#Constants
G=const.G.cgs.value
M_sun=const.M_sun.cgs.value
L_sun=const.L_sun.cgs.value
R_sun=const.R_sun.cgs.value
kb=const.k_B.cgs.value
sigma_sb = const.sigma_sb.cgs.value
mp=const.m_p.cgs.value
me=const.m_e.cgs.value
h=const.h.cgs.value
c=const.c.cgs.value
pc=const.pc.cgs.value
au=const.au.cgs.value
kpc=1.E3*pc
eV=u.eV.to('erg')
#Hubble time (1/H0) th=4.55*10**17 s
th=((Planck13.H0)**-1.0).to('s').value
##Julian year --3.15576E7 s
year=u.yr.to('s')
sigma_sb=const.sigma_sb.cgs.value

