import bilby
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
import precession
from bilby.gw.eos import EOSFamily, TabularEOS
from scipy.interpolate import CubicSpline
from itertools import repeat
from numba import jit
import scipy.special
import scipy.integrate
import scipy.spatial.transform
from numba import njit


import warnings
warnings.filterwarnings("ignore", category=scipy.integrate.IntegrationWarning)
def fxn():
    warnings.warn("IntegrationWarning", scipy.integrate.IntegrationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Or if you are using > Python 3.11:
with warnings.catch_warnings(action="ignore"):
    fxn()



def eval_k_from_lambda(lambda_):
    if np.all(lambda_)==0:
        return 1
    else:   
       ln_k=0.1940+ 0.09163*np.log(lambda_) + 0.04812*np.log(lambda_)**2 - 0.004283*np.log(lambda_)**3+ 0.00012450*np.log(lambda_)**4
        
       return np.exp(ln_k)

def total_mass_from_chirp_mass_q(chirp_mass, q):
    factor = ((1 + q)**2 / q)**(3/5)
    return chirp_mass * factor
    
def gwfrequency_to_pnseparation(theta1, theta2, deltaphi, fGW, q, chi1, chi2, k1,k2, M_msun, PNorder=[0,1,1.5,2]):
    """
    Convert GW frequency (in Hz) to PN orbital separation (in natural units, c=G=M=1). We use the 2PN expression reported in Eq. 4.13 of Kidder 1995, arxiv:gr-qc/9506022.
    
    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    fGW: float
        Gravitational-wave frequency.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    M_msun: float
        Total mass of the binary in solar masses.
    PNorder: array (default: [0,1,1.5,2])
        PN orders considered.
    
    Returns
    -------
    r: float
        Binary separation.
    
    Examples
    --------
    ``r = precession.gwfrequency_to_pnseparation(theta1,theta2,deltaphi,fGW,q,chi1,chi2,M_msun,PNorder=[0,1,1.5,2])``
    """

    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    deltaphi = np.atleast_1d(deltaphi).astype(float)
    fGW = np.atleast_1d(fGW).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)
    M_msun = np.atleast_1d(M_msun).astype(float)

    # Prefactor is pi*Msun*G/c^3/s. It's pi and not 2pi because f is the GW frequency while Kidder's omega is the orbital angular velocity
    tildeomega = M_msun * fGW * 1.5473886176432832e-05 

    r =  tildeomega**(-2/3) * (
        (0 in PNorder) * 1 
        + (1 in PNorder) * tildeomega**(2/3) * (-1/3 * ((1 + q))**(-2) * (3 + q * (5 + 3 * q))) 
        + (1.5 in PNorder) * tildeomega *(-1/3 * ((1 + q))**(-2) * (np.cos(theta1) * (2 + 3 * q) * chi1 + np.cos(theta2)* q * (3 + 2 * q) * chi2))
        + (2 in PNorder) * tildeomega**(4/3) * (1/36 * ((1 + q))**(-4) * (9 * (1 + -3 * (np.cos(theta1))**(2)) * k1 * \
        (chi1)**(3) + q * (171 + (346 * q + (171 * (q)**(2) + (18 * ((1 + \
        q))**(2) * (2 * (np.cos(theta1)) * np.cos(theta2) + -1 * np.cos(deltaphi) * np.sin(theta1) * np.sin(theta2)) * chi1 * \
        chi2 + 9 * (1 + -3 * (np.cos(theta2))**(2)) * k2 * (q)**(3) * (chi2)**(3))))))) 
        )

    return r


def pnseparation_to_gwfrequency(theta1, theta2, deltaphi, r, q, chi1, chi2, k1,k2, M_msun, PNorder=[0,1,1.5,2]):
    """
    Convert PN orbital separation in natural units (c=G=M=1) to GW frequency in Hz. We use the 2PN expression reported in Eq. 4.5 of Kidder 1995, arxiv:gr-qc/9506022.
    
    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    M_msun: float
        Total mass of the binary in solar masses.
    PNorder: array (default: [0,1,1.5,2])
        PN orders considered.
    
    Returns
    -------
    fGW: float
        Gravitational-wave frequency.
    
    Examples
    --------
    ``fGW = precession.pnseparation_to_gwfrequency(theta1,theta2,deltaphi,r,q,chi1,chi2,M_msun,PNorder=[0,1,1.5,2])``
    """


    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    deltaphi = np.atleast_1d(deltaphi).astype(float)
    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    M_msun = np.atleast_1d(M_msun).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)
   
    tildeomega = r**(-3/2) * (
        (0 in PNorder) * 1 
        + (1 in PNorder) * r**(-1) *(-1/2 * ((1 + q))**(-2) * (3 + q * (5 + 3 * q)))
        + (1.5 in PNorder) * r**(-3/2) * (-1/2 * ((1 + q))**(-2) * (np.cos(theta1) * (2 + 3 * q) * chi1 + np.cos(theta2) * q * (3 + 2 * q) * chi2))
        + (2 in PNorder) * r**(-2) * (1/8 * ((1 + q))**(-4) * (15 + (3 * (1 + -3 * (np.cos(theta1))**(2)) * k1 * \
        (chi1)**(3) + q * (107 + (187 * q + (107 * (q)**(2) + (15 * (q)**(3) + \
        (6 * ((1 + q))**(2) * (2 * np.cos(theta1) * np.cos(theta2) + -1 * np.cos(deltaphi) * np.sin(theta1) * np.sin(theta2)) * \
        chi1 * chi2 + 3 * (1 + -3 * (np.cos(theta2)**(2)) * k2 * (q)**(3) * (chi2)**(3)))))))))))

    # Prefactor is pi*Msun*G/c^3/s. It's pi and not 2pi because f is the GW frequency while Kidder's omega is the orbital angular velocity
    fGW = tildeomega / (1.5473886176432832e-05  * M_msun)

    return fGW
    
def eval_chicons(theta1, theta2, r, q, chi1, chi2, k1, k2):
    """
    Generalized effective spin.
    
    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) object: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) object: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.
    
    Returns
    -------
    chicons: float
        Generalized effective spin.
    
    Examples
    --------
    ``chicons = precession.eval_chicons(theta1, theta2, r, q, chi1, chi2, k1, k2)``
    """
    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)

    c1=np.cos(theta1)
    c2=np.cos(theta2)
    return -1 * (r)**(1/2) * (-1 + (((1 + q))**(-2) * (r)**(-1) * (((1 + \
    q))**(2) * r + ((c1)**(2) * k1 * (chi1)**(2) + (2 * c1 * c2 * q * \
    chi1 * chi2 + ((c2)**(2) * k2 * (q)**(2) * (chi2)**(2) + -2 * (1 + q) \
    * (r)**(1/2) * (c1 * chi1 + c2 * q * chi2))))))**(1/2))

def eval_chifake(theta1, theta2, r, q, chi1, chi2, k1, k2):
    """
    Generalized effective spin.
    
    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) object: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) object: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.
    
    Returns
    -------
    chicons: float
        Generalized effective spin.
    
    Examples
    --------
    ``chicons = precession.eval_chicons(theta1, theta2, r, q, chi1, chi2, k1, k2)``
    """
    theta1 = np.atleast_1d(theta1).astype(float)
    theta2 = np.atleast_1d(theta2).astype(float)
    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)

    c1=np.cos(theta1)
    c2=np.cos(theta2)
    return 1/2 * ((1 + q))**(-2) * (r)**(-1/2) * ((c1)**(2) * k1 * (chi1)**(2) + \
    (c2 * q * chi2 * (2 * c1 * chi1 + c2 * k2 * q * chi2) + -2 * ((1 + \
    q))**(2) * r * (-1 + (((1 + q))**(-2) * (r)**(-1) * (((1 + q))**(2) * \
    r + ((c1)**(2) * k1 * (chi1)**(2) + (2 * c1 * c2 * q * chi1 * chi2 + \
    ((c2)**(2) * k2 * (q)**(2) * (chi2)**(2) + -2 * (1 + q) * (r)**(1/2) * \
    (c1 * chi1 + c2 * q * chi2))))))**(1/2))))



def eval_theta1(deltachi, chicons, r, q, chi1, chi2, k1, k2):
    """
    Angle between the orbital angular momentum and the spin of the primary object.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    chicons: float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) object: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) object: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.
    
    
    Returns
    -------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    
    Examples
    --------
    ``theta1 = precession.eval_theta1(deltachi, chicons, r, q, chi1, chi2, k1, k2)``
    """
    costheta1 = eval_costheta1(deltachi, chicons, r, q, chi1, chi2, k1, k2)
    theta1 = np.arccos(costheta1)

    return theta1

def eval_theta2(deltachi, chicons, r, q, chi2, k1, k2):
    """
    Angle between the orbital angular momentum and the spin of the secondary object.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    chicons: float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) object: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.
    
    Returns
    -------
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    
    Examples
    --------
    ``theta2 = precession_NS.eval_theta2(deltachi, chicons, r, q, chi2, k1, k2)``
    """
    costheta2= eval_costheta2(deltachi, chicons, r, q, chi2, k1,k2)
    theta2 = np.arccos(costheta2)
    return theta2

def eval_deltaphi(deltachi, kappa, r, chicons, q, chi1, chi2, k1, k2, cyclesign=1):
    """
    Angle between the projections of the two spins onto the orbital plane.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.    
    cyclesign: integer, optional (default: 1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    
    Returns
    -------
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    
    Examples
    --------
    ``deltaphi = precession.eval_deltaphi(deltachi,kappa,r,chicons,q,chi1,chi2,k1,k2,cyclesign=1)``
    """

    cyclesign = np.atleast_1d(cyclesign)
    cosdeltaphi = eval_cosdeltaphi(deltachi, kappa, r, chicons, q, chi1, chi2, k1, k2)
    deltaphi = np.sign(cyclesign)*np.arccos(cosdeltaphi)

    return deltaphi

def eval_cosdeltaphi(deltachi, kappa, r, chicons, q, chi1, chi2,  k1, k2):
    
    """
    Cosine of the angle between the projections of the two spins onto the orbital plane.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.
    Returns
    -------
    cosdeltaphi: float
        Cosine of the angle between the projections of the two spins onto the orbital plane.
    
    Examples
    --------
    ``cosdeltaphi = precession.eval_cosdeltaphi(deltachi,kappa,r,chicons, q,chi1,chi2, k1,k2)``
    """

    deltachi = np.atleast_1d(deltachi).astype(float)
    kappa = np.atleast_1d(kappa).astype(float)
    r = np.atleast_1d(r).astype(float)
    chicons = np.atleast_1d(chicons).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)

    with warnings.catch_warnings():
        
        # If there are infinitely large separation in the array the following will throw a warning. You can safely ignore it because that value is not used, see below  
        if np.inf in r:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
 

    cosdeltaphi= 1/4 * (q)**(-1) * (((2 + (k1 + k2)))**(-2) * (((2 + (k1 + k2)))**(2) \
    * (chi1)**(2) + -1 * ((1 + q))**(2) * ((2 * (r)**(1/2) + (deltachi + \
    (k2 * deltachi + -1 * ((4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) \
    + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * \
    k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))**(2)))**(-1/2) * (((2 + (k1 + k2)))**(-2) * \
    (((2 + (k1 + k2)))**(2) * (q)**(2) * (chi2)**(2) + -1 * ((1 + \
    q))**(2) * ((-2 * (r)**(1/2) + (deltachi + (k1 * deltachi + ((4 * r + \
    ((1 + -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * \
    (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + \
    (k1 + k2)) * chicons)))))**(1/2)))))**(2)))**(-1/2) * (-2 * \
    ((chi1)**(2) + (q)**(4) * (chi2)**(2)) + (4 * ((2 + (k1 + k2)))**(-1) \
    * q * (1 + q) * (r)**(1/2) * (2 * kappa+ (kappa* k1 + (kappa* k2 + (2 * kappa* q \
    + (kappa* k1 * q + (kappa* k2 * q + (-2 * (r)**(1/2) + (-2 * q * (r)**(1/2) \
    + (-1 * deltachi + (-1 * k2 * deltachi + (q * deltachi + (k1 * q * \
    deltachi + (((4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + ((2 + \
    (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) * \
    deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2) + q * ((4 * r + ((1 \
    + -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) \
    + -2 * (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))))))))))) + 4 * ((2 + (k1 + k2)))**(-2) * q * \
    ((1 + q))**(2) * (-8 * r + (4 * k1 * (r)**(1/2) * deltachi + (-4 * k2 \
    * (r)**(1/2) * deltachi + (k1 * (deltachi)**(2) + (k2 * \
    (deltachi)**(2) + (2 * k1 * k2 * (deltachi)**(2) + (4 * (r)**(1/2) * \
    chicons + (2 * k1 * (r)**(1/2) * chicons + (2 * k2 * (r)**(1/2) * \
    chicons + (-2 * (chicons)**(2) + (-1 * k1 * (chicons)**(2) + (-1 * k2 \
    * (chicons)**(2) + (4 * (r)**(1/2) * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2) \
    + (-1 * k1 * deltachi * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2) \
    + k2 * deltachi * ((4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + \
    ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) \
    * deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2)))))))))))))))))

    cosdeltaphi = np.where(r!=np.inf, cosdeltaphi, np.cos(np.random.uniform(0,np.pi, len(cosdeltaphi))))

    return cosdeltaphi

def eval_costheta1(deltachi, chicons, r, q, chi1, chi2, k1, k2):
    """
    Cosine of the angle between the orbital angular momentum and the spin of the primary object.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    chicons: float
        Generalized effective spin.
    r: float
        Binary separation.    
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) object: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) object: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.
    
    Returns
    -------
    costheta1: float
        Cosine of the angle between orbital angular momentum and primary spin.
    
    Examples
    --------
    ``costheta1 = precession.eval_costheta1(deltachi, chicons, r, q, chi1, chi2, k1, k2)``
    """
    deltachi = np.atleast_1d(deltachi).astype(float)
    chicons = np.atleast_1d(chicons).astype(float)
    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)

    u= precession.eval_u(r=r,q=q)
    return ((2 + (k1 + k2)))**(-1) * (1 + q) * (chi1)**(-1) * (2 * (r)**(1/2) + \
    (deltachi + (k2 * deltachi + -1 * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2))))

def eval_costheta2(deltachi, chicons, r, q, chi2, k1, k2):
    """
    Cosine of the angle between the orbital angular momentum and the spin of the secondary object.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    chicons: float
        Generalized effective spin.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0<=q<=1.
    chi2: float
        Dimensionless spin of the secondary (lighther) object: 0<=chi1<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.
    
    Returns
    -------
    costheta2: float
        Cosine of the angle between orbital angular momentum and secondary spin.
    
    Examples
    --------
    ``costheta2 = precession.eval_costheta2(deltachi, chicons, r, q, chi2, k1, k2)``
    """    

    deltachi = np.atleast_1d(deltachi).astype(float)
    chicons = np.atleast_1d(chicons).astype(float)
    r = np.atleast_1d(r).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)

    return -1 * ((2 + (k1 + k2)))**(-1) * (q)**(-1) * (1 + q) * (chi2)**(-1) * \
    (-2 * (r)**(1/2) + (deltachi + (k1 * deltachi + ((4 * r + ((1 + -1 * \
    k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * \
    (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2))))
      


def conserved_to_angles(deltachi, kappa, r, chicons, q, chi1, chi2, k1,k2, cyclesign=+1):
    """
    Convert conserved quantities (deltachi,kappa,chieff) into angles (theta1,theta2,deltaphi).
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons: float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.
        
    cyclesign: integer, optional (default: +1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    
    Returns
    -------
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    
    Examples
    --------
    ``theta1,theta2,deltaphi = precession.conserved_to_angles(deltachi,kappa,r,chicons,q,chi1,chi2,cyclesign=+1)``
    """


    theta1= eval_theta1(deltachi, chicons,r, q, chi1, chi2, k1, k2)
    theta2 = eval_theta2(deltachi, chicons, r,q, chi2, k1, k2)
    deltaphi = eval_deltaphi(deltachi, kappa, r, chicons, q, chi1, chi2, k1,k2, cyclesign=cyclesign)

    return np.stack([theta1, theta2, deltaphi])

def angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, k1, k2,  full_output=False):
    """
    Convert angles (theta1,theta2,deltaphi) into conserved quantities (deltachi,kappa,chicons).
    
    Parameters
    ----------
    theta1: float
        Angle between orbital angular momentum and primary spin.
    theta2: float
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float
        Angle between the projections of the two spins onto the orbital plane.
    r: float
        Binary separation.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless spin-induced quadrupole of the primary (heavier) object: k1= 1 for BH, k1> 1 for NS.
    k2: float
        Dimensionless spin-induced quadrupole of the secondary (lighter) object: k2= 1 for BH, k2> 1 for NS.    
    full_output: boolean, optional (default: False)
        Return additional outputs.
    
    Returns
    -------
    chicons: float
       Generalized effective spin.
    cyclesign: integer, optional
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``deltachi,kappa,chicons = precession.angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2)``
    ``deltachi,kappa,chicons,cyclesign = precession.angles_to_conserved(theta1,theta2,deltaphi,r,q,chi1,chi2,full_output=True)``
    """


    deltachi = precession.eval_deltachi(theta1, theta2, q, chi1, chi2)
    kappa = precession.eval_kappa(theta1=theta1, theta2=theta2, deltaphi=deltaphi, r=r, q=q, chi1=chi1, chi2=chi2)
    chicons = eval_chicons(theta1, theta2,r,  q, chi1, chi2, k1, k2)

    if full_output:
        cyclesign = np.where(r==np.inf,np.nan,precession.eval_cyclesign(deltaphi=deltaphi))
        return np.stack([deltachi, kappa, chicons, cyclesign])

    else:
        return np.stack([deltachi, kappa, chicons])





def eval_chieff_fromdeltachi(deltachi, chicons, r, k1, k2):
    return 1/2 * ((2 + (k1 + k2)))**(-1) * (8 * (r)**(1/2) + (-2 * k1 * \
    deltachi + (2 * k2 * deltachi + -1 * ((((-8 * (r)**(1/2) + (2 * k1 * \
    deltachi + -2 * k2 * deltachi)))**(2) + -4 * (2 + (k1 + k2)) * (-2 * \
    (deltachi)**(2) + (k1 * (deltachi)**(2) + (k2 * (deltachi)**(2) + (8 * \
    (r)**(1/2) * chicons + -4 * (chicons)**(2)))))))**(1/2))))
def eval_lambda_from_mass(mass, eos):
    """
    Evaluate the tidal deformability from the mass using the EOS family.
    
    Parameters
    ----------
    mass: float
        Mass in solar masses.
    eos_family: EOSFamily
        The EOS family to use for the evaluation.
    
    Returns
    -------
    lambda_: float
        Tidal deformability.
    """
    mpa_eos = TabularEOS(eos)
    mpa_fam = EOSFamily(mpa_eos)
    m_max=mpa_fam.maximum_mass
    if mass > m_max:
        raise ValueError(f"Mass {mass} exceeds maximum mass {m_max} for the EOS family.")
    
    return  mpa_fam.lambda_from_mass(mass)


##@jit(nopython=True, cache=True)
#@njit(fastmath=True)
def dchidt2_RHS(deltachi, kappa, r,  chicons,  q, chi1, chi2, k1,k2): 
        #if dchidt
    """
    Right-hand side of the (ddeltachi/dt)^2 equation. 
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons: float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the primary (heavier) body: k1> 1 NS, k1=1 BH
    
    Returns
    -------
    dchidt2: float
        Squared time derivative of the weighted spin difference.
    
    Examples
    --------
    ``dchidt2 = precession.dchidt2_RHS(deltachi, kappa, r, chicons,  q, chi1, chi2, k1,k2)``
    """

   # q=np.atleast_1d(q).astype(float)
    #r= precession.eval_r(u=u,q=q)
    #
    pre_2=9/4 * (q)**(2) * ((1 + q))**(-4) * (r)**(-7) * (chi1)**(2) * \
    (chi2)**(2) * (4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + ((2 + \
    (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) * \
    deltachi + (2 + (k1 + k2)) * chicons))))

    sin1_2=(1 + -1 * ((2 + (k1 + k2)))**(-2) * ((1 + q))**(2) * (chi1)**(-2) * \
    ((2 * (r)**(1/2) + (deltachi + (k2 * deltachi + -1 * ((4 * r + ((1 + \
    -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + \
    -2 * (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))**(2))

    
    sin2_2=(1 + -1 * ((2 + (k1 + k2)))**(-2) * (q)**(-2) * ((1 + q))**(2) * \
    (chi2)**(-2) * ((-2 * (r)**(1/2) + (deltachi + (k1 * deltachi + ((4 * \
    r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * \
    (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + \
    (k1 + k2)) * chicons)))))**(1/2)))))**(2))


    sinDeltaPhi_2=(1 + -1/16 * ((2 + (k1 + k2)))**(4) * (q)**(-2) * ((((2 + (k1 + \
    k2)))**(2) * (chi1)**(2) + -1 * ((1 + q))**(2) * ((2 * (r)**(1/2) + \
    (deltachi + (k2 * deltachi + -1 * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))**(2)))**(-1) * ((((2 + (k1 + k2)))**(2) * \
    (q)**(2) * (chi2)**(2) + -1 * ((1 + q))**(2) * ((-2 * (r)**(1/2) + \
    (deltachi + (k1 * deltachi + ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))**(2)))**(-1) * ((-2 * ((chi1)**(2) + (q)**(4) \
    * (chi2)**(2)) + (4 * ((2 + (k1 + k2)))**(-1) * q * (1 + q) * \
    (r)**(1/2) * (2 * kappa + (kappa * k1 + (kappa * k2 + (2 * kappa * q + (kappa * k1 * q + \
    (kappa * k2 * q + (-2 * (r)**(1/2) + (-2 * q * (r)**(1/2) + (-1 * \
    deltachi + (-1 * k2 * deltachi + (q * deltachi + (k1 * q * deltachi + \
    (((4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * \
    (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + \
    (k1 + k2)) * chicons)))))**(1/2) + q * ((4 * r + ((1 + -1 * k1 * k2) \
    * (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * \
    (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))))))))))) + 4 * ((2 + (k1 + k2)))**(-2) * q * \
    ((1 + q))**(2) * (-8 * r + (4 * k1 * (r)**(1/2) * deltachi + (-4 * k2 \
    * (r)**(1/2) * deltachi + (k1 * (deltachi)**(2) + (k2 * \
    (deltachi)**(2) + (2 * k1 * k2 * (deltachi)**(2) + (4 * (r)**(1/2) * \
    chicons + (2 * k1 * (r)**(1/2) * chicons + (2 * k2 * (r)**(1/2) * \
    chicons + (-2 * (chicons)**(2) + (-1 * k1 * (chicons)**(2) + (-1 * k2 \
    * (chicons)**(2) + (4 * (r)**(1/2) * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2) \
    + (-1 * k1 * deltachi * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2) \
    + k2 * deltachi * ((4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + \
    ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) \
    * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2))))))))))))))))))**(2))
    dchidt2 =( pre_2 * sin1_2 * sin2_2 * sinDeltaPhi_2)
    #print( pre_2, sin1_2, sin2_2, sinDeltaPhi_2)
    return np.where(dchidt2>0, dchidt2,np.nan)


def dchidt_RHS(deltachi, kappa, chicons, u, q, chi1, chi2, k1,k2):
        #if dchidt
    """
    Right-hand side of the (ddeltachi/dt) equation. 
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons: float
        Genric conserved spin parameter.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the primary (heavier) body: k1> 1 NS, k1=1 BH
    
    Returns
    -------
    dchidt: float
        time derivative of the weighted spin difference.
    
    Examples
    --------
    ``dchidt = precession.dchidt_RHS(r,chieff,q)``
    """
    r= precession.eval_r(u=u,q=q)
    q=np.atleast_1d(q).astype(float)
    pre=-3/2 * q * ((1 + q))**(-3) * (r)**(-7/2) * chi1 * chi2 * ((1 + -1 * \
    ((2 + (k1 + k2)))**(-2) * ((1 + q))**(2) * (chi1)**(-2) * ((2 * \
    (r)**(1/2) + (deltachi + (k2 * deltachi + -1 * ((4 * r + ((1 + -1 * \
    k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * \
    (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))**(2)))**(1/2) * (-2 * (1 + q) * (r)**(1/2) + \
    ((1 + k1) * ((2 + (k1 + k2)))**(-1) * (1 + q) * (2 * (r)**(1/2) + \
    (deltachi + (k2 * deltachi + -1 * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))) + -1 * (1 + k2) * ((2 + (k1 + k2)))**(-1) * (1 \
    + q) * (-2 * (r)**(1/2) + (deltachi + (k1 * deltachi + ((4 * r + ((1 \
    + -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) \
    + -2 * (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))) * ((1 + -1 * ((2 + (k1 + k2)))**(-2) * \
    (q)**(-2) * ((1 + q))**(2) * (chi2)**(-2) * ((-2 * (r)**(1/2) + \
    (deltachi + (k1 * deltachi + ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))**(2)))**(1/2) * ((1 + -1/16 * ((2 + (k1 + \
    k2)))**(4) * (q)**(-2) * ((((2 + (k1 + k2)))**(2) * (chi1)**(2) + -1 \
    * ((1 + q))**(2) * ((2 * (r)**(1/2) + (deltachi + (k2 * deltachi + -1 \
    * ((4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) \
    * (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 \
    + (k1 + k2)) * chicons)))))**(1/2)))))**(2)))**(-1) * ((((2 + (k1 + \
    k2)))**(2) * (q)**(2) * (chi2)**(2) + -1 * ((1 + q))**(2) * ((-2 * \
    (r)**(1/2) + (deltachi + (k1 * deltachi + ((4 * r + ((1 + -1 * k1 * \
    k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * \
    (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))**(2)))**(-1) * ((-2 * ((chi1)**(2) + (q)**(4) \
    * (chi2)**(2)) + (4 * ((2 + (k1 + k2)))**(-1) * q * (1 + q) * \
    (r)**(1/2) * (2 * kappa + (kappa * k1 + (kappa * k2 + (2 * kappa * q + (kappa * k1 * q + \
    (kappa * k2 * q + (-2 * (r)**(1/2) + (-2 * q * (r)**(1/2) + (-1 * \
    deltachi + (-1 * k2 * deltachi + (q * deltachi + (k1 * q * deltachi + \
    (((4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * \
    (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + \
    (k1 + k2)) * chicons)))))**(1/2) + q * ((4 * r + ((1 + -1 * k1 * k2) \
    * (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * \
    (r)**(1/2) * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2)))))))))))))) + 4 * ((2 + (k1 + k2)))**(-2) * q * \
    ((1 + q))**(2) * (-8 * r + (4 * k1 * (r)**(1/2) * deltachi + (-4 * k2 \
    * (r)**(1/2) * deltachi + (k1 * (deltachi)**(2) + (k2 * \
    (deltachi)**(2) + (2 * k1 * k2 * (deltachi)**(2) + (4 * (r)**(1/2) * \
    chicons + (2 * k1 * (r)**(1/2) * chicons + (2 * k2 * (r)**(1/2) * \
    chicons + (-2 * (chicons)**(2) + (-1 * k1 * (chicons)**(2) + (-1 * k2 \
    * (chicons)**(2) + (4 * (r)**(1/2) * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2) \
    + (-1 * k1 * deltachi * ((4 * r + ((1 + -1 * k1 * k2) * \
    (deltachi)**(2) + ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) \
    * ((k1 + -1 * k2) * deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2) \
    + k2 * deltachi * ((4 * r + ((1 + -1 * k1 * k2) * (deltachi)**(2) + \
    ((2 + (k1 + k2)) * (chicons)**(2) + -2 * (r)**(1/2) * ((k1 + -1 * k2) \
    * deltachi + (2 + (k1 + k2)) * \
    chicons)))))**(1/2))))))))))))))))))**(2)))**(1/2)
    return pre 

def deltachiroots(kappa,  u, chicons, q, chi1, chi2, k1, k2, precomputedroots=None):
    """
    Roots of the cubic equation in deltachi that describes the dynamics on the precession timescale.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
    u: float
        Compactified separation 1/(2L).
    chicons: float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the primary (heavier) body: k1> 1 NS, k1=1 BH
    precomputedroots: array, optional (default: None)
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    deltachi-nth: float, optional
        Spurious roots of the deltachi evolution.
    deltachiminusus: float
        Lowest physical root of the deltachi evolution.
    deltachiminus: float
        Lowest physical root of the deltachi evolution.
    
    Examples
    --------
    `all_roots = precession.deltachiroots(kappa, u, chicons, q, chi1, chi2, k1, k2, precomputedroots=precomputedroots)`
    """
    r=precession.eval_r(u=u, q=q)

    if precomputedroots is None:
        grid_size = int(1e4) 
        x = np.linspace(-1, 1, grid_size)
        y = dchidt2_RHS(x, kappa, r, chicons, q, chi1, chi2, k1, k2)
        mask = np.isfinite(y)
        spl = CubicSpline(x[mask],y[mask]/np.max(y[mask]) )
        roots= spl.roots()
       
        roots = roots[(roots >= -1) & (roots <= 1)]


        # vectorized physical filtering
        c1 = eval_costheta1(roots, chicons, r, q, chi1, chi2, k1, k2)
        c2 = eval_costheta2(roots, chicons, r, q, chi2, k1, k2)
      
        mask = (
        np.isfinite(c1) &
        np.isfinite(c2) &
        (np.abs(c1) <= 1) &
        (np.abs(c2) <= 1)
)
       # print(roots[mask])
        return roots[mask]

    else: 
        precomputedroots=np.array(precomputedroots)
    
        return precomputedroots
    


    
def eval_tau(kappa, r, chicons,  q, chi1, chi2, k1, k2, precomputedbounds=None):
    """
    Evaluate the nutation period.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the primary (heavier) body: k1> 1 NS, k1=1 BH
    precomputedbounds: array, optional (default: None)
        Pre-computed output of deltachi_plusminus for computational efficiency.
    Returns
    -------
    tau: float
        Nutation period.
    
    Examples
    --------
    ``tau = precession.eval_tau(kappa, r, chicons,  q, chi1, chi2, k1, k2, precomputedbounds=precomputedbounds)``
    """
    
    q=np.atleast_1d(q).astype(float)
    kappa = np.atleast_1d(kappa).astype(float)
    r = np.atleast_1d(r).astype(float)
    u= precession.eval_u(r=r,q=q)
    chicons = np.atleast_1d(chicons).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)
    roots = deltachi_plusminus(kappa, u, chicons, q, chi1, chi2, k1, k2, precomputedroots=precomputedbounds)
    roots = np.atleast_2d(roots).astype(float)
    
    def _integrand_tau(deltachi, kappa, chicons, r, q, chi1, chi2, k1, k2):
        dchidt2 = dchidt2_RHS(deltachi, kappa, r,  chicons, q, chi1, chi2,k1,k2) 
        return 1 / dchidt2**(1/2)
    
    def _compute(roots, kappa, chicons, r, q, chi1, chi2,k1,k2):
        #print(roots[0],roots[1])
        res,err= scipy.integrate.quad(_integrand_tau,roots[0], roots[1], args=(kappa, chicons, r, q, chi1, chi2,k1,k2), epsrel=1e-12, epsabs=1e-12)
        return res
 
  
    tau = np.array(list(map(_compute, roots, kappa, chicons, r,  q, chi1, chi2, k1, k2)))
    tau = np.array(tau).astype(float)
    #if np.isnan(tau).any():
      #  print('nan found',roots[0], roots[1])
        
       # print(dchidt2_RHS(roots[0][0], kappa, u,  chicons, q, chi1, chi2, k1,k2) )
        #print(dchidt2_RHS(roots[1], kappa, u,  chicons, q, chi1, chi2, k1,k2) )
              
      #  print(roots)
    return 2*tau

def safe_rhs(x, direction, kappa, r, chicons, q, chi1, chi2, k1, k2, max_iter=100):
    """
    Nudge x in the specified direction until dchidt2_RHS(x) becomes positive.
    
    direction:
        +1  -> move upward
        -1  -> move downward
    """

    step = 1e-10 

    for _ in range(max_iter):
        z = dchidt2_RHS(x, kappa, r, chicons, q, chi1, chi2, k1, k2)

        # Accept only strictly positive finite values
        if np.isfinite(z):
            return x, z

        # Otherwise nudge
        x += direction * step
        step += step  # grow step exponentially

    raise RuntimeError("RHS remained invalid after nudging.")

def deltachi_plusminus(kappa, u, chicons, q, chi1, chi2, k1, k2, precomputedroots=None):
    """
    Limits on the weighted spin difference for given (kappa, r, chicons, q, chi1, chi2, k1, k2). These are two of the solutions of the underlying cubic polynomial.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons: float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the primary (heavier) body: k1> 1 NS, k1=1 BH
    
    Returns
    -------
    deltachiplus: float
        Maximum value of the weighted spin difference.
    deltachiminus: float
        Minimum value of the weighted spin difference.
    
    Examples
    --------
    ``deltachiminus,deltachiplus = precession.deltachi_plusminus(kappa,u,chicons,q,chi1,chi2, k1,k2)``
 
    
    q=np.atleast_1d(q).astype(float)
    chicons = np.atleast_1d(chicons).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)
    kappa= np.atleast_1d(kappa).astype(float)
    """
    #all roots which need to be sorted and selected
    r= precession.eval_r(u=u,q=q)
    if precomputedroots is not None:
        valid_roots = np.array(precomputedroots)    
    else:
        valid_roots = np.sort(deltachiroots(kappa, u, chicons, q, chi1, chi2, k1, k2))

    if len(valid_roots) == 0:
        raise ValueError("No valid roots found for the given parameters.")
    elif len(valid_roots) ==2:
        deltachiminus = valid_roots[0]
        deltachiplus = valid_roots[1]
    elif len(valid_roots) == 3:   
        deltachiminus1 = valid_roots[0]
        deltachiplus1 = valid_roots[1]
        if np.isnan(dchidt2_RHS((deltachiplus1+deltachiminus1)/2,kappa,r, chicons,q,chi1,chi2, k1, k2)):
            deltachiminus =valid_roots[1]
            deltachiplus = valid_roots[2]
        else:
            deltachiminus = deltachiminus1
            deltachiplus = deltachiplus1

    elif len(valid_roots) == 4:
        print('4 roots at:', kappa, r,chicons)
        deltachiminus1 = valid_roots[0]
        deltachiplus1 = valid_roots[1]
        tau1= eval_tau( kappa,r, chicons, q, chi1, chi2, k1, k2,precomputedbounds=[deltachiminus1,deltachiplus1]).T[0]
        deltachiminus2 = valid_roots[2]
        deltachiplus2 = valid_roots[3]
        tau2= eval_tau( kappa, r, chicons, q, chi1, chi2, k1, k2,precomputedbounds=[deltachiminus2,deltachiplus2]).T[0]

        eps = np.random.uniform()
        if eps < tau1/(tau1+tau2):
            deltachiminus, deltachiplus = deltachiminus1, deltachiplus1
        else:
            deltachiminus, deltachiplus = deltachiminus2, deltachiplus2   
    else:
        raise ValueError("Unexpected number of valid roots found: {}".format(len(valid_roots)))
    
    #deltachiminusF, z1 = safe_rhs(deltachiminus, 1,kappa, u, chicons, q, chi1, chi2, k1, k2)
    #deltachiplusF,  z2 = safe_rhs(deltachiplus, -1,kappa, u, chicons, q, chi1, chi2, k1, k2)

    return np.stack([deltachiminus, deltachiplus])



def deltachisampling(kappa, r, chicons, q, chi1, chi2, k1, k2, N=1, precomputedroots=None):
    """
    Sample N values of deltachi at fixed separation accoring to its PN-weighted distribution function.
    Can only be used to sample the *same* number of configuration for each binary. If the inputs have shape (M,) the output will have shape
        - (M,N) if M>1 and N>1;
        - (M,) if N=1;
        - (N,) if M=1.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.\
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the secondary (heavier) body: k2> 1 NS, k2=1 BH    
    N: integer, optional (default: 1)
        Number of samples.
    precomputedbounds: array, optional (default: None)
        Pre-computed output of deltachi_plusminus for computational efficiency.
    
    Returns
    -------
    deltachi: float
        Weighted spin difference.
    
    Examples
    --------
    ``deltachi = precession.deltachisampling(kappa,r,chicons,q,chi1,chi2,k1,k2,N=1)``
    """

    u= precession.eval_u(r=r,q=q)

    # Compute the deltachi roots only once and pass them to both functions
    deltachiminus,deltachiplus= deltachi_plusminus(kappa, u, chicons, q, chi1, chi2, k1, k2, precomputedroots=precomputedroots)
                                                         

    tau = eval_tau(kappa, r, chicons, q, chi1, chi2, k1, k2, precomputedbounds=[deltachiminus,deltachiplus])

    # For each binary, generate N samples between 0 and tau.
    # For r=infinity use a simple placeholder
    
   
    t = np.random.uniform(np.zeros(len(tau)),np.where(r!=np.inf, tau, 0),size=(N,len(tau)))


    # np.squeeze is necessary to return shape (M,) instead of (M,1) if N=1
    # np.atleast_1d is necessary to return shape (1,) instead of (,) if M=N=1
    t= np.atleast_1d(np.squeeze(t))

    # Note the special broadcasting rules of deltachioft, see deltachioft.__docs__
    # deltachi has shape (M, N).
    deltachi = deltachioft(t, kappa, r, chicons, q, chi1, chi2, k1,k2, precomputedroots=np.stack([deltachiminus,deltachiplus]),precomputedtau=tau )
    # For infinity use the analytic result. Ignore q=1 "divide by zero" warning:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        #add function to gro from chicons to chieff
        theta1=eval_theta1(deltachi,chicons, r, q, chi1, chi2, k1, k2)
        theta2=eval_theta2(deltachi,chicons, r, q,  chi2, k1, k2) 
        chieff=precession.eval_chieff(theta1,theta2,q,chi1,chi2) 
        deltachiinf = np.squeeze( precession.eval_deltachiinf(kappa, chieff, q, chi1, chi2))
    
    deltachi=np.where(r!=np.inf, deltachi,deltachiinf)
    return np.squeeze(deltachi.T)




###@jit(nopython=True, cache=True)
def dydt(t, y, kappa,  r,chicons, q, chi1, chi2, k1, k2):
    return (dchidt2_RHS(y[0], kappa, r, chicons, q, chi1, chi2, k1,k2))**(1/2)

def deltachioft(t, kappa, r, chicons, q, chi1, chi2, k1, k2, precomputedroots=None, precomputedtau=None, intial_deltachi=None, initial_time=0):
    """
    Evolution of deltachi on the precessional timescale (without radiation reaction).
    The broadcasting rules for this function are more general than those of the rest of the code. The variable t is allowed to have shapes (N,M) while all the other variables have shape (N,). This is useful to sample M precession configuration for each of the N binaries specified as inputs.
    
    Parameters
    ----------
    t: float
        Time.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.\
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the secondary (lighter) body: k2> 1 NS, k2=1 BH 
    precomputedroots: array, optional (default: None)
        Pre-computed output of deltachiroots for computational efficiency.
    precomputedtao: float, optional (default: None)
        Pre-computed output of eval_tau for computational efficiency.
    Returns
    -------
    deltachi: float
        Weighted spin difference.
    
    Examples
    --------
    ``deltachi = precession.deltachioft(t,kappa,r,chicons,q,chi1,chi2, k1,k2)``
    """
    t = np.atleast_1d(t).astype(float)
    u = precession.eval_u(r,q)

    if precomputedroots is None:
        deltachiminus, deltachiplus = deltachi_plusminus(kappa, u, chicons,  q, chi1, chi2, k1, k2, precomputedroots=precomputedroots)
    
    else:
        deltachiminus, deltachiplus = precomputedroots

    if precomputedtau is not None:
        tau = precomputedtau
    else:
        tau = eval_tau(kappa, r, chicons, q, chi1, chi2, k1, k2, precomputedbounds=np.stack([deltachiminus, deltachiplus]))
    if intial_deltachi is not None:
        if intial_deltachi < deltachiminus or intial_deltachi > deltachiplus:
            raise ValueError("Initial deltachi value out of bounds.")
        # Integrate from the initial value to t
        sol1 =scipy.integrate.solve_ivp(dydt, (initial_time, tau/2), [intial_deltachi], args=(kappa, r, chicons, q, chi1, chi2, k1, k2),method='LSODA', dense_output=True, rtol=1e-10, atol=1e-10)
    else:    
        sol1 =scipy.integrate.solve_ivp(dydt, (0, tau/2), [deltachiminus+1e-10], args=(kappa, r, chicons, q, chi1, chi2, k1, k2),method='LSODA', dense_output=True, rtol=1e-10, atol=1e-10)
    #print(sol1)
    
    t_bounded = np.mod(t, tau)
    deltachi = np.empty_like(t_bounded)

    mask1 = t_bounded < tau/2
    mask2 = (t_bounded > tau/2) & (t_bounded < tau)
    mask3 = t_bounded == tau/2
    mask4 = t_bounded == 0

    if np.any(mask1):
        deltachi[mask1] = sol1.sol(t_bounded[mask1])[0]
    if np.any(mask2):
        xtrans = tau - t_bounded[mask2]
        deltachi[mask2] = sol1.sol(xtrans)[0]
    if np.any(mask3):
        deltachi[mask3] = deltachiplus
    if np.any(mask4):
        deltachi[mask4] = deltachiminus

    return  deltachi#, t_bounded


###@jit(nopython=True, cache=True)
def dtdy(y,t,  kappa, chicons, r, q, chi1, chi2, k1, k2):
    return  1/(dchidt2_RHS(y, kappa, r, chicons, q, chi1, chi2, k1,k2))**(1/2)

def tofdeltachi(deltachi, kappa, r, chicons, q, chi1, chi2, k1, k2, precomputedroots=None):
    """
    Time as a function of deltachi on the precessional timescale (without radiation reaction).
    Time is always defined within half of a single precession period, i.e. 0 <= t < tau/2. Then the value of deltachi repet themself

    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.\
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the secondary (lighter) body: k2> 1 NS, k2=1 BH 
    precomputedroots: array, optional (default: None)
        Pre-computed output of deltachiroots for computational efficiency.
    
    Returns
    -------
    deltachi: float
        Weighted spin difference.
        
    
    Examples
    --------
    ``deltachi = precession.deltachioft(t,kappa,r,chicons,q,chi1,chi2, k1,k2)``
    """

    u = precession.eval_u(r,q)

    if precomputedroots is None:
        deltachiminus, deltachiplus = deltachi_plusminus(kappa,u, chicons, q, chi1, chi2, k1, k2, precomputedroots=precomputedroots)
    else:
        deltachiminus, deltachiplus = precomputedroots

   
    sol1 =scipy.integrate.solve_ivp(dtdy, (deltachiminus+1e-6, deltachiplus-1e-6), [0], args=(kappa, chicons, r, q, chi1, chi2, k1, k2),method='LSODA', dense_output=True, rtol=1e-10, atol=1e-10)
   
    return  sol1.sol(deltachi)[0]

def precession_average(kappa, r, chicons, q, chi1, chi2, k1, k2, func, *args, precomputedroots=None, method='quadrature', Nsamples=1e4):
    """
    Average a generic function over a precession cycle. The function needs to have call: func(deltachi, *args). Keywords arguments are not supported.
    There are integration methods implemented:
        - method='quadrature' uses scipy.integrate.quad. This is set by default and should be preferred.
        - method='montecarlo' samples t(deltachi) and approximate the integral with a Monte Carlo sum. The number of samples can be specifed by Nsamples.
    Additional keyword arguments are passed to scipy.integrate.quad.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the secondary (lighter) body: k2> 1 NS, k2=1 BH     
    func: function
        Function to precession-average.
    *args: tuple
        Extra arguments to pass to func.
    method: string (default: 'quadrature')
        Either 'quadrature' or 'montecarlo'
    Nsamples: integer, optional (default: 1e4)
        Number of Monte Carlo samples.
    
    Returns
    -------
    func_av: float
        Precession average of func.
    
    Examples
    --------
    ``func_av = precession.precession_average(kappa,r,chieff,q,chi1,chi2,k1,k2,func,*args,method='quadrature',Nsamples=1e4)``
    """

    kappa=np.atleast_1d(kappa).astype(float)
    r=np.atleast_1d(r).astype(float)
    chicons=np.atleast_1d(chicons).astype(float)
    q=np.atleast_1d(q).astype(float)
    chi1=np.atleast_1d(chi1).astype(float)
    chi2=np.atleast_1d(chi2).astype(float)
    k1=np.atleast_1d(k1).astype(float)
    k2=np.atleast_1d(k2).astype(float)
    u = precession.eval_u(r=r,q=q)
    deltachiminus,deltachiplus= deltachi_plusminus(kappa, u, chicons, q, chi1, chi2,k1, k2, precomputedroots=precomputedroots)
    if method == 'quadrature':

        tau = eval_tau(kappa, r, chicons, q, chi1, chi2, k1,k2, precomputedbounds=np.stack([deltachiminus,deltachiplus]))

        # Each args needs to be iterable
        args = [np.atleast_1d(a) for a in args]

        # Compute the numerator explicitely

        def _integrand(deltachi, kappa, r, chicons, q, chi1, chi2, k1, k2, *args):
            dchidt2 = dchidt2_RHS(deltachi, kappa, r, chicons, q, chi1, chi2, k1, k2)
            return func(deltachi, *args) / dchidt2**(1/2)

        def _compute(deltachiminus,deltachiplus, kappa, r, chicons, q, chi1, chi2, k1, k2, *args):
            res, err =  scipy.integrate.quad(_integrand, deltachiminus, deltachiplus, args=(kappa, r, chicons, q, chi1, chi2, k1,k2, *args))
            return res
        
        func_av = np.array(list(map(_compute, [deltachiminus], [deltachiplus], kappa, r, chicons, q, chi1, chi2, k1,k2,*args))) / tau * 2 

    elif method == 'montecarlo':
        
        deltachi = deltachisampling(kappa, r, chicons, q, chi1, chi2, k1,k2, N=int(Nsamples), precomputedroots=np.stack([deltachiminus,deltachiplus]))
        evals = func(deltachi, *args)
        func_av = np.sum(evals, axis=-1)/Nsamples
        func_av = np.atleast_1d(func_av)

    else:
        raise ValueError("Available methods are 'quadrature' and 'montecarlo'.")

    return func_av

#@njit(fastmath=True)
def dchiconsdu_rhs(deltachi, u, chicons, q, k1, k2):
    return ((2 + (k1 + k2)))**(-1) * (q)**(-1) * ((1 + q))**(2) * (u)**(-2) * \
    ((((1 + q))**(2) + -2 * q * u * chicons))**(-1) * (-2 + (-2 * (q)**(2) \
    + (q * (-4 + ((k1 + -1 * k2) * u * deltachi + (2 + (k1 + k2)) * u * \
    chicons)) + 2 * q * u * (((q)**(-2) * ((1 + q))**(4) * (u)**(-2) + \
    ((1 + -1 * k1 * k2) * (deltachi)**(2) + ((2 + (k1 + k2)) * \
    (chicons)**(2) + -1 * (q)**(-1) * ((1 + q))**(2) * (u)**(-1) * ((k1 + \
    -1 * k2) * deltachi + (2 + (k1 + k2)) * chicons)))))**(1/2))))

#@njit(fastmath=True)
def dkdu_rhs(deltachi, kappa, u, chicons, q, k1, k2):
    """
    Right-hand side of the dSsquared/du ODE describing precession-averaged inspiral.
    This is an internal function used by the ODE integrator and is not array-compatible.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    u: float
        Compactified separation 1/(2L).
    chicons: float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the secondary (lighter)) body: k2 > 1 NS, k2=1 BH
    
    Returns
    -------
    RHS: float
        Right-hand side.
    
    Examples
    --------
    ``RHS = precession.Ssquared_rhs(deltachi,kappa,u,chicons,q,k1,k2)``
    """
 
    return ((2 + (k1 + k2)))**(-1) * (q)**(-1) * ((1 + q))**(-1) * (u)**(-2) * \
    (-1 * ((1 + q))**(3) + (q * (-1 + (-1 * k2 + (q + k1 * q))) * u * \
    deltachi + ((2 + (k1 + k2)) * q * (1 + q) * u * kappa + (1 + q) * ((1 \
    + q * (4 + ((-1 * k1 + k2) * u * deltachi + (q * (6 + (4 * q + \
    ((q)**(2) + (-1 * (k1 + -1 * k2) * (2 + q) * u * deltachi + (1 + -1 * \
    k1 * k2) * (u)**(2) * (deltachi)**(2))))) + (-1 * (2 + (k1 + k2)) * \
    ((1 + q))**(2) * u * chicons + (2 + (k1 + k2)) * q * (u)**(2) * \
    (chicons)**(2)))))))**(1/2))))

def dkdu_rhs_precav(kappa, u, chicons, q, chi1, chi2, k1, k2, precomputedroots=None):
    """
    Right-hand side of the dkappa/du ODE describing precession-averaged inspiral.
    This is an internal function used by the ODE integrator and is not array-compatible.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
    u: float
        Compactified separation 1/(2L).
    chicons: float
       Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the secondary (lighter)) body: k2 > 1 NS, k2=1 BH

    Returns
    -------
    RHS: float
        Right-hand side.
    
    Examples
    --------
    ``RHS = precession.rhs_precav(kappa,u,chicons,q,chi1,chi2,k1,k2)``
    """

    if u <= 0:
       # In this case use analytic result
        if q==1: # TODO: think about this again
            Ssav = (chi1**2+q**4 * chi2**2)/(1 + q)**4  #- ( 2*q*(kappa*(1+q) -chieff)*(kappa*(1+q) -q*chieff)/((-1 + q)**2 *(1 + q)**2))
        else:#MUST BE RECHECKED!
            Ssav = (chi1**2+q**4 * chi2**2)/(1 + q)**4  - ( 2*q*(kappa*(1+q) -chicons)*(kappa*(1+q) -q*chicons)/((1-q)**2 *(1 + q)**2))

    else:
            args=np.array([ np.squeeze(kappa), np.squeeze(u), np.squeeze(chicons), np.squeeze(q), k1, k2])
            r=precession.eval_r(u=u, q=q)
            Ssav = precession_average(kappa, r, chicons, q, chi1, chi2,k1,k2, dkdu_rhs, *args, precomputedroots=precomputedroots, method='quadrature', Nsamples=1000)

    # The output is always real, even if the input is complex
    if np.iscomplex(np.squeeze(Ssav)):
        Ssav = np.real(np.squeeze(Ssav))

    return (np.squeeze(Ssav))

def dchiconsdu_rhs_precav(kappa, u, chicons, q, chi1, chi2,  k1, k2, precomputedroots=None):
    """
    Right-hand side of the dkappa/du ODE describing precession-averaged inspiral.
    This is an internal function used by the ODE integrator and is not array-compatible.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
    u: float
        Compactified separation 1/(2L).
    chicons: float
       Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the secondary (lighter)) body: k2 > 1 NS, k2=1 BH

    Returns
    -------
    RHS: float
        Right-hand side.
    
    Examples
    --------
    ``RHS = precession.rhs_precav(kappa,u,chicons,q,chi1,chi2,k1,k2)``
    """


    args=np.array([ np.squeeze( u), np.squeeze(chicons), q, k1, k2])
    r=precession.eval_r(u=u, q=q)
    rhs_chicons_precav= precession_average(kappa, r, chicons, q, chi1, chi2,k1,k2, dchiconsdu_rhs, *args, precomputedroots=precomputedroots, method='quadrature', Nsamples=1000)

    # The output is always real, even if the input is complex
    if np.iscomplex(np.squeeze(rhs_chicons_precav)):
        rhs_chicons_precav = np.real(np.squeeze(rhs_chicons_precav))

    return (np.squeeze(rhs_chicons_precav))


def integrator_precav(kappainitial, u, chicons, q, chi1, chi2, k1, k2, **odeint_kwargs):
    """
    Integration of ODE dkappa/du describing precession-averaged inspirals. Here u needs to be an array with lenght >=1, where u[0] corresponds to the initial condition and u[1:] corresponds to the location where outputs are returned. For past time infinity use u=0.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    kappainitial: float
        Initial value of the regularized momentum kappa.
    u: float
        Compactified separation 1/(2L).
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the secondary (lighter)) body: k2 > 1 NS, k2=1 BH

    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``kappa = precession.integrator_precav(kappainitial,u,chieff,q,chi1,chi2)``
    """

    kappainitial = np.atleast_1d(kappainitial).astype(float)
    u = np.atleast_2d(u).astype(float)
    chicons = np.atleast_1d(chicons).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)
    # Defaults for the integrators, can be changed by the user

    if 'rol' not in odeint_kwargs: odeint_kwargs['rtol']=1e-10
    if 'aol' not in odeint_kwargs: odeint_kwargs['atol']=1e-10
    
    # I'm sorry but this needs to be forced for compatibility with the rest of the code

    def _system(u,vec, q, chi1, chi2, k1, k2 ):
        kappa, chicons= vec # Wrapper for the solver
        deltachiminus,deltachiplus= deltachi_plusminus(kappa, u, chicons, q, chi1, chi2,k1, k2)
        return [dkdu_rhs_precav(kappa, u, chicons, q, chi1, chi2, k1, k2, precomputedroots=np.stack([deltachiminus,deltachiplus])),dchiconsdu_rhs_precav(kappa, u, chicons, q, chi1, chi2,  k1, k2, precomputedroots=np.stack([deltachiminus,deltachiplus]))]
    
    def _compute(kappainitial, chiconsinitial, u, q, chi1, chi2, k1, k2, odeint_kwargs):
        solution = scipy.integrate.solve_ivp(_system,[u[0], u[-1]], [kappainitial, chiconsinitial], t_eval=u, args=(q, chi1, chi2, k1, k2),method='DOP853', **odeint_kwargs)
      
        ys=solution.y
        
        ODEsolution = np.array([ys[0], ys[1]])
        return ODEsolution 
    
    ODEsolution = np.array(list(map(_compute, kappainitial, chicons, u, q, chi1, chi2, k1, k2, repeat(odeint_kwargs))))
    
    return ODEsolution

def inspiral_precav(theta1=None, theta2=None, deltaphi=None, kappa=None, r=None, u=None, chicons=None, q=None, chi1=None, chi2=None, k1=None, k2=None, requested_outputs=None, enforce=False, **odeint_kwargs):
    """
    Perform precession-averaged inspirals. The variables q, chi1, and chi2 must always be provided.
    The integration range must be specified using either r or u (and not both). These need to be arrays with lenght >=1, where e.g. r[0] corresponds to the initial condition and r[1:] corresponds to the location where outputs are returned. For past time infinity use either u=0 or r=np.inf.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    The initial conditions must be specified in terms of one an only one of the following:
        - theta1,theta2, and deltaphi (but note that deltaphi is not necessary if integrating from infinite separation).
        - kappa, chieff.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'].
    The flag enforce allows checking the consistency of the input variables.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
    r: float, optional (default: None)
        Binary separation.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    requested_outputs: list, optional (default: None)
        Set of outputs.
    enforce: boolean, optional (default: False)
        If True raise errors, if False raise warnings.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_precav(kappa,u=u,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """

    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, kappa, r, u, chicons, q, chi1, chi2, k1, k2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(precession.tiler(None, np.atleast_1d(q))))
        else:
            if k == 4 or k == 5:  # Either u or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, kappa, r, u, chicons, q, chi1, chi2, k1, k2 = inputs

    # This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'deltachiminus', 'deltachiplus', 'chicons', 'chieff','q', 'chi1', 'chi2', 'k1', 'k2'])
    # If in doubt, return everything
    if requested_outputs is None:
        requested_outputs = alloutputs

    def _compute(theta1, theta2, deltaphi, kappa, r, u, chicons, q, chi1, chi2, k1, k2):

        # Make sure you have q, chi1, and chi2.
        if q is None or chi1 is None or chi2 is None or k1 is None or k2 is None:
            raise TypeError("Please provide q, chi1, chi2 and k1, k2.")

        # Make sure you have either r or u.
        if r is not None and u is None:
            assert np.logical_or(precession.ismonotonic(r, '<='), precession.ismonotonic(r, '>=')), 'r must be monotonic'
            u = precession.eval_u(r, precession.tiler(q, r))
        elif r is None and u is not None:
            assert np.logical_or(precession.ismonotonic(u, '<='), precession.ismonotonic(u, '>=')), 'u must be monotonic'
            r = precession.eval_r(u=u, q=precession.tiler(q, u))
        else:
            raise TypeError("Please provide either r or u. Use np.inf for infinity.")

        assert np.sum(u == 0) <= 1 and np.sum(u[1:-1] == 0) == 0, "There can only be one r=np.inf location, either at the beginning or at the end."

        # User provided theta1,theta2, and deltaphi. Get chieff and kappa.
        if theta1 is not None and theta2 is not None and deltaphi is not None and kappa is None and chicons is None:
            deltachi, kappa, chicons = angles_to_conserved(theta1, theta2, deltaphi, r[0], q, chi1, chi2, [k1], [k2])
        # User provides kappa, chieff
        elif theta1 is None and theta2 is None and deltaphi is None and kappa is not None and chicons is not None:
            pass

        else:
            raise TypeError("Please provide one and not more of the following: (theta1,theta2,deltaphi), (kappa, chicons).")

        # Enforce limits. Uncomment if you want to be more restrictive (though some integrations are going to fail for roundoff errors in the resonant finder)
        """
        if enforce:
            chieffmin, chieffmax = chiefflimits(q, chi1, chi2)
            assert chieff >= chieffmin and chieff <= chieffmax, "Unphysical initial conditions [inspiral_precav]."+str(theta1)+" "+str(theta2)+" "+str(deltaphi)+" "+str(kappa)+" "+str(r)+" "+str(u)+" "+str(chieff)+" "+str(q)+" "+str(chi1)+" "+str(chi2)
            kappamin,kappamax = kappalimits(r=r[0], chieff=chieff, q=q, chi1=chi1, chi2=chi2)
            assert kappa >= kappamin and kappa <= kappamax, "Unphysical initial conditions [inspiral_precav]."+str(theta1)+" "+str(theta2)+" "+str(deltaphi)+" "+str(kappa)+" "+str(r)+" "+str(u)+" "+str(chieff)+" "+str(q)+" "+str(chi1)+" "+str(chi2)
        """
        # Actual integration.
        kappa, chicons = np.squeeze(integrator_precav(kappa, u, chicons, q, chi1, chi2, [k1], [k2], **odeint_kwargs))
        chieff=None
        deltachiminus = None
        deltachiplus = None
        deltachi=None
        theta1=None
        theta2=None
        deltaphi=None

        # Roots along the evolution
        if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi', 'deltachi', 'deltachiminus', 'deltachiplus', 'chieff']):
            deltachiminus,deltachiplus= np.array(list(map(deltachi_plusminus,kappa, u, chicons, precession.tiler(q,r),precession.tiler(chi1,r),precession.tiler(chi2,r),precession.tiler(k1,r),precession.tiler(k2,r)))).T
            # Resample deltachi
            if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi', 'deltachi']):
                deltachi = np.array(list(map(deltachisampling, kappa, r, chicons,  precession.tiler(q,r),precession.tiler(chi1,r),precession.tiler(chi2,r),precession.tiler(k1,r),precession.tiler(k2,r)))).T
                #deltachi=[]
                #for kk, rr, cc, qq, c1, c2, kk1, kk2, roots in zip(kappa,r, chicons,precession.tiler(q,r), precession.tiler(chi1,r),precession.tiler(chi2,r), precession.tiler(k1,r),precession.tiler(k2,r), np.stack([deltachiminus,deltachiplus], axis=-1)):
                 #    deltachi.append(deltachisampling(kk, rr, cc, qq, c1, c2, kk1, kk2,precomputedroots=roots, N=int(1)))
#

                #deltachi=np.array(deltachi).squeeze()
                if any(x in requested_outputs for x in ['theta1', 'theta2', 'deltaphi']):
                    theta1,theta2,deltaphi = conserved_to_angles(deltachi, kappa, r,chicons, precession.tiler(q,r),precession.tiler(chi1,r),precession.tiler(chi2,r),precession.tiler(k1,r),precession.tiler(k2,r), cyclesign = np.random.choice([-1, 1], r.shape))
                    if any(x in requested_outputs for x in ['chieff']):
                     chieff = precession.eval_chieff(theta1, theta2,  precession.tiler(q,r),precession.tiler(chi1,r),precession.tiler(chi2,r)).T
        #print(theta1, theta2, deltaphi, deltachi, kappa, r, u, deltachiminus, deltachiplus, chicons,chieff, q, chi1, chi2, k1, k2)
        return theta1, theta2, deltaphi, deltachi, kappa, r, u, deltachiminus, deltachiplus, chicons,chieff, q, chi1, chi2, k1, k2

    # Here I force dtype=object because the outputs have different shapes
    allresults = np.array(list(map(_compute, theta1, theta2, deltaphi, kappa, r, u, chicons, q, chi1, chi2, k1,k2)), dtype=object).T

    # Return only requested outputs (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs, requested_outputs)

    # Store into a dictionary
    outcome = {}

    for k, v in zip(alloutputs[wantoutputs], allresults[wantoutputs]):
        outcome[k] = np.squeeze(np.stack(v))

        # For the constants of motion...
        if  k == 'q' or k == 'chi1' or k == 'chi2' or k == 'k1' or k == 'k2':  # Constants of motion
            outcome[k] = np.atleast_1d(outcome[k])
        #... and everything else
        else:
            outcome[k] = np.atleast_2d(outcome[k])

    return outcome




def deltachi_plusminus_vec(kappa, u, chicons, q, chi1, chi2, k1, k2, precomputedroots=None):
    """
    Vectorized version of deltachi_plusminus.
    Works on arrays of parameters.
    """
    kappa = np.atleast_1d(kappa)
    u = np.atleast_1d(u)
    chicons = np.atleast_1d(chicons)
    q = np.atleast_1d(q)
    chi1 = np.atleast_1d(chi1)
    chi2 = np.atleast_1d(chi2)
    k1 = np.atleast_1d(k1)
    k2 = np.atleast_1d(k2)
    shape = kappa.shape
    deltamin = np.full(shape, np.nan)
    deltaplus = np.full(shape, np.nan)
    # flatten for easier iteration (still faster than nested loops)
    for idx in np.ndindex(shape):
        kval, uval, chic, qval, chi1v, chi2v, k1v, k2v = (
            kappa[idx], u[idx], chicons[idx], q[idx], chi1[idx], chi2[idx], k1[idx], k2[idx]
        )

        if precomputedroots is not None:
            valid_roots = np.array(precomputedroots)
        else:
            valid_roots = np.sort(deltachiroots(kval, uval, chic, qval, chi1v, chi2v, k1v, k2v))

        if len(valid_roots) == 0:
            continue
        elif len(valid_roots) == 2:
            deltamin[idx], deltaplus[idx] = valid_roots
        elif len(valid_roots) == 3:
            dmin1, dplus1 = valid_roots[0], valid_roots[1]
            tau1 = eval_tau(kval, precession.eval_r(u=uval,q=qval), chic, qval, chi1v, chi2v, k1v, k2v,
                            precomputedbounds=[dmin1, dplus1]).T[0]
            if np.isfinite(tau1):
                deltamin[idx], deltaplus[idx] = dmin1, dplus1
            else:
                deltamin[idx], deltaplus[idx] = valid_roots[1], valid_roots[2]
        elif len(valid_roots) == 4:
            dmin1, dplus1 = valid_roots[0], valid_roots[1]
            dmin2, dplus2 = valid_roots[2], valid_roots[3]
            tau1 = eval_tau(kval, precession.eval_r(u=uval,q=qval), chic, qval, chi1v, chi2v, k1v, k2v,
                            precomputedbounds=[dmin1, dplus1]).T[0]
            tau2 = eval_tau(kval, precession.eval_r(u=uval,q=qval), chic, qval, chi1v, chi2v, k1v, k2v,
                            precomputedbounds=[dmin2, dplus2]).T[0]
            eps = np.random.uniform()
            if eps < tau1/(tau1+tau2):
                deltamin[idx], deltaplus[idx] = dmin1, dplus1
            else:
                deltamin[idx], deltaplus[idx] = dmin2, dplus2
        else:
            continue

    return np.stack([deltamin, deltaplus])



def deltachi_plusminusOLD(kappa, u, chicons, q, chi1, chi2, k1, k2, precomputedroots=None):
    """
    Limits on the weighted spin difference for given (kappa, r, chicons, q, chi1, chi2, k1, k2). These are two of the solutions of the underlying cubic polynomial.
    
    Parameters
    ----------
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chicons: float
        Generalized effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    k1: float
        Dimensionless quadruple parameter of the primary (heavier) body: k1> 1 NS, k1=1 BH
    k2: float
        Dimensionless quadruple paraemters of the primary (heavier) body: k1> 1 NS, k1=1 BH
    
    Returns
    -------
    deltachiplus: float
        Maximum value of the weighted spin difference.
    deltachiminus: float
        Minimum value of the weighted spin difference.
    
    Examples
    --------
    ``deltachiminus,deltachiplus = precession.deltachi_plusminus(kappa,r,chicons,q,chi1,chi2, k1,k2)``
    """
    r=precession.eval_r(u=u, q=q)
    q=np.atleast_1d(q).astype(float)
    u= precession.eval_u(r=r,q=q)
    chicons = np.atleast_1d(chicons).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)

    #all roots which need to be sorted and selected
    if precomputedroots is not None:
        valid_roots = np.array(precomputedroots)    
    else:
        valid_roots = np.sort(deltachiroots(kappa, u, chicons, q, chi1, chi2, k1, k2))
        #print("Valid roots:", valid_roots)

    if len(valid_roots) == 0:
        raise ValueError("No valid roots found for the given parameters.")
    elif len(valid_roots) ==2:
        deltachiminus = valid_roots[0]
        deltachiplus = valid_roots[1]
    elif len(valid_roots) == 3:   
        deltachiminus1 = valid_roots[0]
        deltachiplus1 = valid_roots[1]
        tau1= eval_tau(kappa, r, chicons, q, chi1, chi2, k1, k2, precomputedbounds=[deltachiminus1,deltachiplus1]).T[0]
        if np.isfinite(tau1):
            deltachiminus = deltachiminus1
            deltachiplus = deltachiplus1
        else:
            deltachiminus =valid_roots[1]
            deltachiplus = valid_roots[2]

    elif len(valid_roots) == 4:
        deltachiminus1 = valid_roots[0]
        deltachiplus1 = valid_roots[1]
        tau1= eval_tau( kappa, r, chicons, q, chi1, chi2, k1, k2,precomputedbounds=[deltachiminus1,deltachiplus1]).T[0]
        deltachiminus2 = valid_roots[2]
        deltachiplus2 = valid_roots[3]
        tau2= eval_tau( kappa, r, chicons, q, chi1, chi2, k1, k2,precomputedbounds=[deltachiminus2,deltachiplus2]).T[0]
       # print("tau1, tau2:", tau1, tau2)
        epsilon=np.random.uniform(0,1)
        if epsilon < tau1/(tau1+tau2):
            deltachiminus = deltachiminus1
            deltachiplus = deltachiplus1
        elif epsilon >  tau1/(tau1+tau2) and epsilon < 1:
            deltachiminus = deltachiminus2
            deltachiplus = deltachiplus2
        else:
            raise ValueError("Unexpected epsilon value: {}".format(epsilon))    
    else:
        raise ValueError("Unexpected number of valid roots found: {}".format(len(valid_roots)))


    """
    # Correct when too close to perfect alignment
    angleup=tiler(0,q)
    angledown=tiler(np.pi,q)

    chieffupup = eval_chieff(angleup, angleup, q, chi1, chi2)
    deltachiupup = eval_deltachi(angleup, angleup, q, chi1, chi2)
    deltachiminusus = np.where(np.isclose(chieff,chieffupup), deltachiupup,deltachiminusus)
    deltachiminus = np.where(np.isclose(chieff,chieffupup), deltachiupup,deltachiminus)

    chieffdowndown = eval_chieff(angledown, angledown, q, chi1, chi2)
    deltachidowndown = eval_deltachi(angledown, angledown, q, chi1, chi2)
    deltachiminusus = np.where(np.isclose(chieff,chieffdowndown), deltachidowndown,deltachiminusus)
    deltachiminus = np.where(np.isclose(chieff,chieffdowndown), deltachidowndown,deltachiminus)
    """
    return np.stack([deltachiminus, deltachiplus])


################ Orbit-averaged evolution ################

def rhs_orbav(allvars, v, q, m1, m2, eta, chi1, chi2, k1, k2, S1, S2, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3,3.5]):
    """
    Right-hand side of the systems of ODEs describing orbit-averaged inspiral. The equations are reported in Sec 4A of Gerosa and Kesden, arXiv:1605.01067. The format is d[allvars]/dv=RHS where allvars=[Lhx,Lhy,Lhz,S1hx,S1hy,S1hz,S2hx,S2hy,S2hz,t], h indicates unit vectors, v is the orbital velocity, and t is time.
    This is an internal function used by the ODE integrator and is not array-compatible.
    
    Parameters
    ----------
    allvars: array
        Packed ODE input variables.
    v: float
        Newtonian orbital velocity.
    q: float
        Mass ratio: 0<=q<=1.
    m1: float
        Mass of the primary (heavier) black hole.
    m2: float
        Mass of the secondary (lighter) black hole.
    eta: float
        Symmetric mass ratio 0<=eta<=1/4.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    S1: float
        Magnitude of the primary spin.
    S2: float
        Magnitude of the secondary spin.
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    
    Returns
    -------
    RHS: float
        Right-hand side.
    
    Examples
    --------
    ``RHS = precession.rhs_orbav(allvars,v,q,m1,m2,eta,chi1,chi2,S1,S2,PNorderpre=[0,0.5],PNorderrad=[0,1,1.5,2,2.5,3,3.5])``
    """

    # Unpack inputs
    Lh = allvars[0:3]
    S1h = allvars[3:6]
    S2h = allvars[6:9]
    t = allvars[9]

    # Angles
    ct1 = np.dot(S1h, Lh)
    ct2 = np.dot(S2h, Lh)
    ct12 = np.dot(S1h, S2h)

    # Spin precession for S1    # MUST CHANGE!!1 done
    Omega1 = (0 in PNorderpre) * eta*v**5*(2+3*q/2)*Lh + (0.5 in PNorderpre) * v**6*(S2*S2h-3*S2*ct2*Lh-3*q*S1*k1*ct1*Lh)/2
    dS1hdt = np.cross(Omega1, S1h)

    # Spin precession for S2 # MUST CHANGE!!1 done
    Omega2 = (0 in PNorderpre) * eta*v**5*(2+3/(2*q))*Lh + (0.5 in PNorderpre) * v**6*(S1*S1h-3*S1*ct1*Lh-3*S2*ct2*k2*Lh/q)/2
    dS2hdt = np.cross(Omega2, S2h)

    # Conservation of angular momentum # WHAT ABOUT THIS??? done
    dLhdt = -v*(S1*dS1hdt+S2*dS2hdt)/eta

    dvdt = (32*eta*v**9/5) * (
        + (0 in PNorderrad) * 1
        - (1 in PNorderrad)*v**2 
                 * (743+924*eta)/336
        + (1.5 in PNorderrad) * v**3 
                 * (4*np.pi
                 - chi1*ct1*(113*m1**2/12 + 25*eta/4)
                 - chi2*ct2*(113*m2**2/12 + 25*eta/4))
        + (2 in PNorderrad) * v**4 
                 * (34103/18144 + 13661*eta/2016 + 59*eta**2/18
                 + eta*chi1*chi2 * (721*ct1*ct2 - 247*ct12)/48
                 + ((m1*chi1)**2 * ((7 + (-240 * k1 + (ct1)**(2) * (-1 + 720 * k1)))))/96 #main changes
                 + ((m2*chi2)**2 * ((7 + (-240 * k2 + (ct2)**(2) * (-1 + 720 * k2)))))/96) #main changes
        - (2.5 in PNorderrad) * v**5 
                 * np.pi*(4159+15876*eta)/672
        + (3 in PNorderrad)*v**6 
                 * (16447322263/139708800 + 16*np.pi**2/3
                 - 1712*(0.5772156649+np.log(4*v))/105
                 + (451*np.pi**2/48 - 56198689/217728)*eta
                 + 541*eta**2/896 - 5605*eta**3/2592)
        + (3.5 in PNorderrad) * v**7 
                 * np.pi*(-4415/4032 + 358675*eta/6048
                 + 91495*eta**2/1512))

    # Integrate in v, not in time
    dtdv = 1./dvdt
    dLhdv = dLhdt*dtdv
    dS1hdv = dS1hdt*dtdv
    dS2hdv = dS2hdt*dtdv

    # Pack outputs
    return np.concatenate([dLhdv, dS1hdv, dS2hdv, [dtdv]])


def integrator_orbav(Lhinitial, S1hinitial, S2hinitial, v, q, chi1, chi2,k1,k2, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3,3.5], **odeint_kwargs):
    """
    Integration of the systems of ODEs describing orbit-averaged inspirals.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    Lhinitial: array
        Initial direction of the orbital angular momentum, unit vector.
    S1hinitial: array
        Initial direction of the primary spin, unit vector.
    S2hinitial: array
        Initial direction of the secondary spin, unit vector.
    v: float
        Newtonian orbital velocity.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    ODEsolution: array
        Solution of the ODE.
    
    Examples
    --------
    ``ODEsolution = precession.integrator_orbavintegrator_orbav(Lhinitial,S1hinitial,S2hinitial,v,q,chi1,chi2)``
    """

    Lhinitial = np.atleast_2d(Lhinitial).astype(float)
    S1hinitial = np.atleast_2d(S1hinitial).astype(float)
    S2hinitial = np.atleast_2d(S2hinitial).astype(float)
    v = np.atleast_2d(v).astype(float)
    q = np.atleast_1d(q).astype(float)
    chi1 = np.atleast_1d(chi1).astype(float)
    chi2 = np.atleast_1d(chi2).astype(float)
    k1 = np.atleast_1d(k1).astype(float)
    k2 = np.atleast_1d(k2).astype(float)

    # Defaults for the integrators, can be changed by the user
    if 'mxstep' not in odeint_kwargs: odeint_kwargs['mxstep']=5000000
    if 'rol' not in odeint_kwargs: odeint_kwargs['rtol']=1e-10
    if 'aol' not in odeint_kwargs: odeint_kwargs['atol']=1e-10
    odeint_kwargs['full_output']=0 # This needs to be forced for compatibility with the rest of the code

    def _compute(Lhinitial, S1hinitial, S2hinitial, v, q, chi1, chi2, k1,k2):

        # I need unit vectors
        print(np.isclose(np.linalg.norm(Lhinitial), 1), np.isclose(np.linalg.norm(S1hinitial), 1),np.isclose(np.linalg.norm(S2hinitial), 1))
        assert np.isclose(np.linalg.norm(Lhinitial), 1)
        assert np.isclose(np.linalg.norm(S1hinitial), 1)
        assert np.isclose(np.linalg.norm(S2hinitial), 1)
        
        # Pack inputs
        ic = np.concatenate([Lhinitial, S1hinitial, S2hinitial, [0]])

        # Compute these quantities here instead of inside the RHS for speed
        m1 = precession.eval_m1(q).item()
        m2 = precession.eval_m2(q).item()
        S1 = precession.eval_S1(q, chi1).item()
        S2 = precession.eval_S2(q, chi2).item()
        eta = precession.eval_eta(q).item()

        # solve_ivp implementation. Didn't really work.
        #ODEsolution = scipy.integrate.solve_ivp(rhs_orbav, (vinitial, vfinal), ic, method='LSODA', t_eval=(vinitial, vfinal), dense_output=True, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula),rtol=1e-12,atol=1e-12)
        #ODEsolution = scipy.integrate.solve_ivp(rhs_orbav, (vinitial, vfinal), ic, t_eval=(vinitial, vfinal), dense_output=True, args=(q, m1, m2, eta, chi1, chi2, S1, S2, quadrupole_formula))

        # Make sure the first step is large enough. This is to avoid LSODA to propose a tiny step which causes the integration to stall
        if 'h0' not in odeint_kwargs: odeint_kwargs['h0']=  np.sign(v[-1]-v[0]) *  v[0]/1e6

        ODEsolution = scipy.integrate.odeint(rhs_orbav, ic, v, args=(q, m1, m2, eta, chi1, chi2,k1,k2, S1, S2, PNorderpre, PNorderrad), **odeint_kwargs)#, printmessg=0,rtol=1e-10,atol=1e-10)#,tcrit=sing)
        return ODEsolution

    ODEsolution = np.array(list(map(_compute, Lhinitial, S1hinitial, S2hinitial, v, q, chi1, chi2 ,k1, k2)))

    return ODEsolution


def inspiral_orbav(theta1=None, theta2=None, deltaphi=None, Lh=None, S1h=None, S2h=None, deltachi=None, kappa=None, r=None, u=None, chicons=None, q=None, chi1=None, chi2=None, k1=None, k2=None, cyclesign=+1, PNorderpre=[0,0.5], PNorderrad=[0,1,1.5,2,2.5,3,3.5], requested_outputs=None, **odeint_kwargs):
    """
    Perform precession-averaged inspirals. The variables q, chi1, and chi2 must always be provided.
    The integration range must be specified using either r or u (and not both). These need to be arrays with lenght >=1, where e.g. r[0] corresponds to the initial condition and r[1:] corresponds to the location where outputs are returned.
    The function is vectorized: evolving N multiple binaries with M outputs requires kappainitial, chieff, q, chi1, chi2 to be of shape (N,) and u of shape (M,N).
    The initial conditions must be specified in terms of one an only one of the following:
        - Lh, S1h, and S2h
        - theta1,theta2, and deltaphi.
        - deltachi, kappa, chieff, cyclesign.
    The desired outputs can be specified with a list e.g. requested_outputs=['theta1','theta2','deltaphi']. All the available variables are returned by default. These are: ['theta1', 'theta2', 'deltaphi', 'deltachi', 'kappa', 'r', 'u', 'deltachiminus', 'deltachiplus', 'deltachi3', 'chieff', 'q', 'chi1', 'chi2'].
    The flag enforce allows checking the consistency of the input variables.
    Additional keywords arguments are passed to `scipy.integrate.odeint` after some custom-made default settings.
    
    Parameters
    ----------
    theta1: float, optional (default: None)
        Angle between orbital angular momentum and primary spin.
    theta2: float, optional (default: None)
        Angle between orbital angular momentum and secondary spin.
    deltaphi: float, optional (default: None)
        Angle between the projections of the two spins onto the orbital plane.
    Lh: array, optional (default: None)
        Direction of the orbital angular momentum, unit vector.
    S1h: array, optional (default: None)
        Direction of the primary spin, unit vector.
    S2h: array, optional (default: None)
        Direction of the secondary spin, unit vector.
    deltachi: float, optional (default: None)
        Weighted spin difference.
    kappa: float, optional (default: None)
        Asymptotic angular momentum.
    r: float, optional (default: None)
        Binary separation.
    u: float, optional (default: None)
        Compactified separation 1/(2L).
    chieff: float, optional (default: None)
        Effective spin.
    q: float, optional (default: None)
        Mass ratio: 0<=q<=1.
    chi1: float, optional (default: None)
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float, optional (default: None)
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: +1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    PNorderpre: array (default: [0,0.5])
        PN orders considered in the spin-precession equations.
    PNorderrad: array (default: [0,0.5])
        PN orders considered in the radiation-reaction equation.
    requested_outputs: list, optional (default: None)
        Set of outputs.
    **odeint_kwargs: unpacked dictionary, optional
        Additional keyword arguments.
    
    Returns
    -------
    outputs: dictionary
        Set of outputs.
    
    Examples
    --------
    ``outputs = precession.inspiral_orbav(Lh=Lh,S1h=S1h,S2h=S2h,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(Lh=Lh,S1h=S1h,S2h=S2h,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,r=r,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(theta1=theta1,theta2=theta2,deltaphi=deltaphi,u=u,q=q,chi1=chi1,chi2=chi2)``
    ``outputs = precession.inspiral_orbav(deltachi=deltachi,kappa=kappa,r=r,chieff=chieff,q=q,chi1=chi1,chi2=chi2)``
    """


    # Substitute None inputs with arrays of Nones
    inputs = [theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chicons, q, chi1, chi2, k1,k2]
    for k, v in enumerate(inputs):
        if v is None:
            inputs[k] = np.atleast_1d(np.squeeze(precession.tiler(None, np.atleast_1d(q))))
        else:
            if k == 3 or k == 4 or k == 5 or k == 8 or k == 9:  # Lh, S1h, S2h, u, or r
                inputs[k] = np.atleast_2d(inputs[k])
            else:  # Any of the others
                inputs[k] = np.atleast_1d(inputs[k])
    theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chicons, q, chi1, chi2, k1,k2 = inputs

    def _compute(theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chicons, q, chi1, chi2, k1,k2, cyclesign):

        if q is None or chi1 is None or chi2 is None or k1 is None or k2 is None:
            raise TypeError("Please provide q, chi1, and chi2 and k1, k2.")

        if r is not None and u is None:
            assert np.logical_or(precession.ismonotonic(r, '<='), precession.ismonotonic(r, '>=')), 'r must be monotonic'
            u = precession.eval_u(r, precession.tiler(q, r))
        elif r is None and u is not None:
            assert np.logical_or(precession.ismonotonic(u, '<='), precession.ismonotonic(u, '>=')), 'u must be monotonic'
            r = precession.eval_r(u=u, q=precession.tiler(q, u))
        else:
            raise TypeError("Please provide either r or u.")

        # User provides Lh, S1h, and S2h
        if Lh is not None and S1h is not None and S2h is not None and theta1 is None and theta2 is None and deltaphi is None and deltachi is None and kappa is None and chicons is None:
            pass

        # User provides theta1, theta2, and deltaphi.
        elif Lh is None and S1h is None and S2h is None and theta1 is not None and theta2 is not None and deltaphi is not None and deltachi is None and kappa is None and chicons is None:
            Lh, S1h, S2h = precession.angles_to_Jframe(theta1, theta2, deltaphi, r[0], q, chi1, chi2)


        # User provides deltachi, kappa, and chieff.
        elif Lh is None and S1h is None and S2h is None and theta1 is None and theta2 is None and deltaphi is None and deltachi is not None and kappa is not None and chicons is not None:
            # cyclesign=+1 by default
            Lh, S1h, S2h = conserved_to_Jframe(deltachi, kappa, r[0], chicons, q, chi1, chi2,k1,k2, cyclesign=cyclesign)
        else:
            raise TypeError("Please provide one and not more of the following: (Lh,S1h,S2h), (theta1,theta2,deltaphi), (deltachi,kappa,chieff).")

        # Make sure vectors are normalized
        Lh = Lh/np.linalg.norm(Lh)
        S1h = S1h/np.linalg.norm(S1h)
        S2h = S2h/np.linalg.norm(S2h)

        v = precession.eval_v(r)

        # Integration
        evaluations = integrator_orbav(Lh, S1h, S2h, v, q, chi1, chi2,k1,k2, PNorderpre=PNorderpre, PNorderrad=PNorderrad,**odeint_kwargs)[0].T
        # For solve_ivp implementation
        #evaluations = np.squeeze(ODEsolution.item().sol(v))

        # Returned output is
        # Lx, Ly, Lz, S1x, S1y, S1z, S2x, S2y, S2z, (t)
        Lh = evaluations[0:3, :].T
        S1h = evaluations[3:6, :].T
        S2h = evaluations[6:9, :].T
        t = evaluations[9, :]

        # Renormalize. The normalization is not enforced by the integrator, it is only maintaied within numerical accuracy.
        #Lh = Lh/np.linalg.norm(Lh)
        #S1h = S1h/np.linalg.norm(S1h)
        #S2h = S2h/np.linalg.norm(S2h)

        S1 = precession.eval_S1(q, chi1)
        S2 = precession.eval_S2(q, chi2)
        L = precession.eval_L(r, precession.tiler(q, r))
        Lvec = (L*Lh.T).T
        S1vec = S1*S1h
        S2vec = S2*S2h
        theta1, theta2, deltaphi = precession.vectors_to_angles(Lvec, S1vec, S2vec)
        deltachi, kappa, chicons, cyclesign = vectors_to_conserved(Lvec, S1vec, S2vec, precession.tiler(q,r),precession.tiler(k1,r),precession.tiler(k2,r), full_output=True)

        return t, theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chicons, q, chi1, chi2, k1, k2, cyclesign

    # This array has to match the outputs of _compute (in the right order!)
    alloutputs = np.array(['t', 'theta1', 'theta2', 'deltaphi', 'Lh', 'S1h', 'S2h', 'deltachi', 'kappa', 'r', 'u', 'chicons', 'q', 'chi1', 'chi2','k1','k2', 'cyclesign'])


    if cyclesign ==+1 or cyclesign==-1:
        cyclesign=np.atleast_1d(precession.tiler(cyclesign,q))
    
    # Here I force dtype=object because the outputs have different shapes
    allresults = np.array(list(map(_compute, theta1, theta2, deltaphi, Lh, S1h, S2h, deltachi, kappa, r, u, chicons, q, chi1, chi2,k1,k2, cyclesign)), dtype=object).T

    # Handle the outputs.
    # Return all
    if requested_outputs is None:
        requested_outputs = alloutputs
    # Return only those requested (in1d return boolean array)
    wantoutputs = np.in1d(alloutputs, requested_outputs)

    # Store into a dictionary
    outcome = {}
    for k, v in zip(alloutputs[wantoutputs], allresults[wantoutputs]):
        outcome[k] = np.squeeze(np.stack(v))

        if k == 'q' or k == 'chi1' or k == 'chi2' or k == 'k2' or k == 'k1':  # Constants of motion (chieff is not enforced!)
            outcome[k] = np.atleast_1d(outcome[k])
        else:
            outcome[k] = np.atleast_2d(outcome[k])

    return outcome



################ Conversions ################



def conserved_to_Lframe(deltachi, kappa, r, chicons, q, chi1, chi2, k1,k2, cyclesign=+1):
    """
    Convert the conserved quanties (deltachi,kappa,chieff) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the orbital angular momentum. In particular, we set Lx=Ly=S1y=0.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: +1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    
    Returns
    -------
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    
    Examples
    --------
    ``Lvec,S1vec,S2vec = precession.conserved_to_Lframe(deltachi,kappa,r,chieff,q,chi1,chi2,cyclesign=+1)``
    """

    theta1,theta2,deltaphi = conserved_to_angles(deltachi, kappa, r, chicons, q, chi1, chi2,k1,k2, cyclesign=cyclesign)
    Lvec, S1vec, S2vec = precession.angles_to_Lframe(theta1, theta2, deltaphi, r, q, chi1, chi2)

    return np.stack([Lvec, S1vec, S2vec])


def conserved_to_Jframe(deltachi, kappa, r, chicons, q, chi1, chi2,k1,k2, cyclesign=+1):
    """
    Convert the conserved quanties (deltachi,kappa,chieff) to angular momentum vectors (L,S1,S2) in the frame
    aligned with the total angular momentum. In particular, we set Jx=Jy=Ly=0.
    
    Parameters
    ----------
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    r: float
        Binary separation.
    chieff: float
        Effective spin.
    q: float
        Mass ratio: 0<=q<=1.
    chi1: float
        Dimensionless spin of the primary (heavier) black hole: 0<=chi1<=1.
    chi2: float
        Dimensionless spin of the secondary (lighter) black hole: 0<=chi2<=1.
    cyclesign: integer, optional (default: +1)
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    
    Returns
    -------
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    
    Examples
    --------
    ``Lvec,S1vec,S2vec = precession.conserved_to_Jframe(deltachi,kappa,r,chieff,q,chi1,chi2,cyclesign=+1)``
    """

    theta1,theta2,deltaphi = conserved_to_angles(deltachi, kappa, r, chicons, q, chi1, chi2,k1,k2, cyclesign=cyclesign)
    Lvec, S1vec, S2vec = precession.angles_to_Jframe(theta1, theta2, deltaphi, r, q, chi1, chi2)

    return np.stack([Lvec, S1vec, S2vec])


def vectors_to_conserved(Lvec, S1vec, S2vec, q,k1,k2,full_output=False):
    """
    Convert vectors (L,S1,S2) to conserved quanties (deltachi,kappa,chieff).
    
    Parameters
    ----------
    Lvec: array
        Cartesian vector of the orbital angular momentum.
    S1vec: array
        Cartesian vector of the primary spin.
    S2vec: array
        Cartesian vector of the secondary spin.
    q: float
        Mass ratio: 0<=q<=1.
    full_output: boolean, optional (default: False)
        Return additional outputs.
    
    Returns
    -------
    chieff: float
        Effective spin.
    cyclesign: integer, optional
        Sign (either +1 or -1) to cover the two halves of a precesion cycle.
    deltachi: float
        Weighted spin difference.
    kappa: float
        Asymptotic angular momentum.
    
    Examples
    --------
    ``deltachi,kappa,chieff = precession.vectors_to_conserved(Lvec,S1vec,S2vec,q)``
    ``deltachi,kappa,chieff,cyclesign = precession.vectors_to_conserved(Lvec,S1vec,S2vec,q,full_output=True)``
    """

    L = precession.norm_nested(Lvec)
    S1 = precession.norm_nested(S1vec)
    S2 = precession.norm_nested(S2vec)

    r = precession.eval_r(L=L,q=q)
    chi1 = precession.eval_chi1(q,S1)
    chi2 = precession.eval_chi2(q,S2)

    theta1,theta2,deltaphi = precession.vectors_to_angles(Lvec, S1vec, S2vec)

    deltachi, kappa, chicons, cyclesign= angles_to_conserved(theta1, theta2, deltaphi, r, q, chi1, chi2, k1,k2,full_output=True)

    if full_output:
        return np.stack([deltachi, kappa, chicons, cyclesign])

    else:
        return np.stack([deltachi, kappa, chicons])

