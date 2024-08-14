#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import contextlib
import os
import matplotlib
from matplotlib.ticker import MaxNLocator
if not(os.environ.has_key('DISPLAY')):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import AxesGrid
import itertools
import lum
import math
import mpmath
import multiprocessing
import numpy as np
import pmap
import pickle
import pdb
import astropy.io.fits as fits
#import pyqt_fit.kde
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import scipy.stats
import time
import util

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

# Avoid excessive space around markers in legend
matplotlib.rcParams['legend.handlelength'] = 0

# Treatment of numpy errors
np.seterr(all='warn')

# Global parameters
par = {'progName': 'jswml.py', 'version': 1.1, 'ev_model': 'z',
       'clean_photom': True, 'use_wt': True, 'kc_use_poly': True}
cosmo = None
sel_dict = {}
chol_par_name = ('alpha', '   M*', ' phi*', ' beta', '  mu*', 'sigma')
methods = ('lfchi', 'denchi', 'post', 'min_slope', 'zero_slope')
mass_limits = (8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12)
mass_zlimits = (0.15, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12)
mag_limits = (-23, -22, -21, -20, -19, -18, -17, -16, -15)
wmax = 5.0  # max incompleteness weighting

# Constants
lg2pi = math.log10(2 * math.pi)
ln10 = math.log(10)
J3 = 30000.0

# Jacknife regions are 4 deg segments starting at given RA
njack = 9
ra_jack = (129, 133, 137, 174, 178, 182, 211.5, 215.5, 219.5)

# Determined from GAMA-I.  Early/late cut at n = 1.9
Q_all = 1.59
P_all = 0.14
Q_early = 1.31
P_early = 0.14
Q_late = 1.98
P_late = 0.13

# Determined from GAMA-II.
Q_clr = {'c': 0.78, 'b': 0.23, 'r': 0.83}
P_clr = {'c': 1.72, 'b': 3.55, 'r': 1.10}
Qdef, Pdef = 1.0, 1.0

# Factor by which to multiply apparent radius in arcsec to get
# absolute radius in kpc when distance measured in Mpc
radfac = math.pi/180.0/3.6

# Solar magnitudes from Blanton et al 2003 for ^{0.1}ugriz bands
Msun_ugriz = [6.80, 5.45, 4.76, 4.58, 4.51]

# FNugrizYJHK (z0=0) Solar magnitudes from Driver et al 2012
Msun_z0 = {'F': 16.02, 'N': 10.18, 'u': 6.38, 'g': 5.15, 'r': 4.71,
           'i': 4.56, 'z': 4.54, 'Y': 4.52, 'J': 4.57, 'H': 4.71, 'K': 5.19}

# Imaging completeness from Blanton et al 2005, ApJ, 631, 208, Table 1
# Modified to remove decline at bright end and to prevent negative
# completeness values at faint end
sb_tab = (18, 19, 19.46, 19.79, 20.11, 20.44, 20.76, 21.09, 21.41,
          21.74, 22.06, 22.39, 22.71, 23.04, 23.36, 23.69, 24.01,
          24.34, 26.00)
comp_tab = (1.0, 1.0, 0.99, 0.97, 0.98, 0.98, 0.98, 0.97, 0.96, 0.96,
            0.97, 0.94, 0.86, 0.84, 0.76, 0.63, 0.44, 0.33, 0.01)

# Polynomial fits to mass completeness limits from misc.mass_comp()
mass_comp_pfit = {'c': (50.96, -57.42, 23.57, 7.32),
                  'b': (44.40, -51.90, 22.22, 7.21),
                  'r': (25.88, -32.11, 15.62, 8.13)}

# Standard symbol and colour order for plots
sym_list = ('ko', 'bs', 'g^', 'r<', 'mv', 'y>', 'cp')
clr_list = 'bgrck'

# Plot labels
mag_petro_label = r'$^{0.1}M_{r_{\rm Petro}} -\ 5 \lg h$'
mag_sersic_label = r'$^{0.1}M_{r_{\rm Sersic}} -\ 5 \lg h$'
den_mag_label = r'$\phi(M)\ (h^3 {\rm Mpc}^{-3} {\rm mag}^{-1})$'
den_mass_label = r'$\phi(M)\ (h^3 {\rm Mpc}^{-3} {\rm dex}^{-1})$'

#------------------------------------------------------------------------------
# Main procedures
#------------------------------------------------------------------------------

def simcat(infile='kcorr.fits', outfile='jswml_sim.fits',
           alpha=-1.23, Mstar=-20.70, phistar=0.01, Q=0.7, P=1.8, chi2max=10,
           Mrange=(-24, -12), mrange=(10, 19.8), zrange=(0.002, 0.65), nz=65,
           fbad=0.03, do_kcorr=True, area_fac=1.0, nblock=500000, schec_nz=0):
    """Generate test data for jswml - see Cole (2011) Sec 5."""

    def gam_dv(z):
        """Gamma function times volume element to integrate."""
        M1 = mrange[1] - cosmo.dist_mod(z) - kcorr(z, pc_med) + Q*(z-par['z0'])
        M1 = max(min(Mrange[1], M1), Mrange[0])
        M2 = mrange[0] - cosmo.dist_mod(z) - kcorr(z, pc_med) + Q*(z-par['z0'])
        M2 = max(min(Mrange[1], M2), Mrange[0])
        L1 = 10**(0.4*(Mstar - M1))
        L2 = 10**(0.4*(Mstar - M2))
        dens = (phistar * 10**(0.4*P*(z-par['z0'])) *
                mpmath.gammainc(alpha+1, L1, L2))
        ans = area * cosmo.dV(z) * dens
        return ans

    def schec(M):
        """Schechter function."""
        L = 10**(0.4*(Mstar - M))
        ans = 0.4 * ln10 * phistar * L**(alpha+1) * np.exp(-L)
        return ans

    def schec_ev(M, z):
        """Evolving Schechter function."""
        L = 10**(0.4*(Mstar - Q*(z-par['z0']) - M))
        ans = 0.4 * ln10 * phistar * L**(alpha+1) * np.exp(-L)
        return ans

    def vol_ev(z):
        """Volume element multiplied by density evolution."""
        pz = cosmo.dV(z) * 10**(0.4*P*(z-par['z0']))
        return pz

    def zM_pdf(z, M):
        """PDF for joint redshift-luminosity distribution.

        Don't use this.  Generate z and M distributions separately."""
        pz = cosmo.dV(z) * 10**(0.4*P*(z-par['z0']))
        pM = schec_ev(M, z)
        return pz*pM

    # Read k-corrections and survey params from input file
    hdulist = fits.open(infile)
    header = hdulist[1].header
    H0 = 100.0
    omega_l = header['OMEGA_L']
    par['z0'] = header['Z0']
    area_dg2 = area_fac*header['AREA']
    area = area_dg2*(math.pi/180.0)*(math.pi/180.0)
    cosmo = util.CosmoLookup(H0, omega_l, zrange)
#    rmin, rmax = cosmo.dm(zrange[0]), cosmo.dm(zrange[1])

    if do_kcorr:
        tbdata = hdulist[1].data
        sel = ((tbdata.field('survey_class') > 3) *
               (tbdata.field('z_tonry') >= zrange[0]) *
               (tbdata.field('z_tonry') < zrange[1]) *
               (tbdata.field('nQ') > 2) * (tbdata['chi2'] < chi2max))
        for ic in xrange(5):
            sel *= np.isfinite(tbdata.field('pcoeff_r')[:, ic])
        tbdata = tbdata[sel]
        nk = len(tbdata)
        ra_gal = tbdata.field('ra')
        pc = tbdata.field('pcoeff_r').transpose()
        pdim = (pc.shape)[1]  # number of coeffs
        hdulist.close()

        # For median k-correction, find median of K(z) and fit poly to this,
        # rather than taking median of coefficients
        zbin = np.linspace(zrange[0], zrange[1], 50)
        k_array = np.polynomial.polynomial.polyval(zbin, pc)
        kmin = np.min(k_array)
        kmax = np.max(k_array)
        print 'kmin, kmax =', kmin, kmax
#        pdb.set_trace()
        k_median = np.median(k_array, axis=0)
        pc_med = np.polynomial.polynomial.polyfit(zbin, k_median, pdim-1)
        k_fit = np.polynomial.polynomial.polyval(zbin, pc_med)
        plt.clf()
        plt.plot(zbin + par['z0'], k_median)
        plt.plot(zbin + par['z0'], k_fit, '--')
        plt.xlabel('z')
        plt.ylabel('K(z)')
        plt.draw()
    else:
        nk = 1
        ra_gal = np.zeros(1)
        pdim = 5
        pc = np.zeros((nk, pdim))
        pc_med = np.zeros(pdim)

    # zbins = np.linspace(zrange[0], zrange[1], nz+1)
    # V_int = area/3.0 * cosmo.dm(zbins)**3
    # V = np.diff(V_int)
    # zcen = zbins[:-1] + 0.5 * (zbins[1]-zbins[0])
    # hist_gen = np.zeros(nz)

    # Predicted N(z) from integrating evolving Schechter function
    if schec_nz:
        zlims = np.linspace(0.0, 0.65, 66)
        nz = len(zlims) - 1
        fout = open(schec_nz, 'w')
        print >> fout, 'alpha {}, M* {}, phi* {}, Q {}, P {}'.format(
                alpha, Mstar, phistar, Q, P)
        for i in xrange(nz):
            zlo = zlims[i]
            zhi = zlims[i+1]
            schec_num, err = scipy.integrate.quad(
                gam_dv, zlo, zhi, epsabs=1e-3, epsrel=1e-3)
            print >> fout, 0.5*(zlo+zhi), schec_num
        fout.close()
        return

    # Integrate evolving LF for number of simulated galaxies
    nsim, err = scipy.integrate.quad(gam_dv, zrange[0], zrange[1],
                                     epsabs=1e-3, epsrel=1e-3)
    nsim = int(nsim)
    print 'Simulating', nsim, 'galaxies'
    mapp_out = np.zeros(nsim)
    Mabs_out = np.zeros(nsim)
    z_out = np.zeros(nsim)
    ra_out = np.zeros(nsim)
    kc_out = np.zeros(nsim)
    pc_out = np.zeros((nsim, 5))

    nrem = nsim
    nout = 0
    while nrem > 0:
        # z, Mabs = util.ran_fun2(zM_pdf, zrange[0], zrange[1], 
        #                         Mrange[0], Mrange[1], nblock)
        z = util.ran_fun(vol_ev, zrange[0], zrange[1], nblock)
        Mabs = util.ran_fun(schec, Mrange[0], Mrange[1], nblock) - Q*(z-par['z0'])

        # Calculate apparent mag and test for visibility
        # First do a crude cut without k-corrections
        mapp = Mabs + cosmo.dist_mod(z)
        sel = (mapp >= mrange[0] - kmax) * (mapp < mrange[1] - kmin)
        z, Mabs, mapp = z[sel], Mabs[sel], mapp[sel]
        nsel = len(z)

        # K-corrections for remaining objects
        kidx = np.random.randint(0, nk, nsel)
        pc_ran = pc[kidx, :]
        kc = np.array(map(lambda i: kcorr(z[i], pc_ran[i, :]), xrange(nsel)))
        mapp += kc
        ra = ra_gal[kidx]
        sel = (mapp >= mrange[0]) * (mapp < mrange[1])
        z, Mabs, mapp, ra, kc, pc_ran = z[sel], Mabs[sel], mapp[sel], ra[sel], kc[sel], pc_ran[sel, :]
        nsel = len(z)
        if nsel > nrem:
            nsel = nrem
            z, Mabs, mapp, ra, kc, pc_ran = z[:nrem], Mabs[:nrem], mapp[:nrem], ra[:nrem], kc[:nrem], pc_ran[:nrem, :]

        mapp_out[nout:nout+nsel] = mapp
        Mabs_out[nout:nout+nsel] = Mabs
        z_out[nout:nout+nsel] = z
        ra_out[nout:nout+nsel] = ra
        kc_out[nout:nout+nsel] = kc
        pc_out[nout:nout+nsel, :] = pc_ran

        nout += nsel
        nrem -= nsel
        print nrem

    # Randomly resample in redshift bins to induce density fluctuations
    zbins = np.linspace(zrange[0], zrange[1], nz+1)
    V_int = area/3.0 * cosmo.dm(zbins)**3
    V = np.diff(V_int)
    zcen = zbins[:-1] + 0.5 * (zbins[1]-zbins[0])
    zhist, bin_edges = np.histogram(z_out, bins=zbins)
    hist_gen = np.zeros(nz)
    # mapp_samp = []
    # Mabs_samp = []
    # z_samp = []
    # ra_samp = []
    # kc_samp = []
    # pc_samp = []
    samp_list = []
    nsamp = 0
    print 'iz  delta  zhist  nsel'
    for iz in xrange(nz):
        zlo = zbins[iz]
        zhi = zbins[iz+1]
        delta = np.random.normal(0.0, math.sqrt(J3/V[iz]))
        nsel = int(round((1+delta) * zhist[iz]))
        hist_gen[iz] = nsel
        print iz, delta, zhist[iz], nsel
        if nsel > 0:
            sel = (zlo <= z_out) * (z_out < zhi)
            idx = np.where(sel)
            samp = np.random.randint(0, zhist[iz], nsel)
            for i in xrange(nsel):
                samp_list.append(idx[0][samp[i]])
                # mapp_samp.append(mapp_out[idx][samp[i]])
                # Mabs_samp.append(Mabs_out[idx][samp[i]])
                # z_samp.append(z_out[idx][samp[i]])
                # ra_samp.append(ra_out[idx][samp[i]])
                # kc_samp.append(kc_out[idx][samp[i]])
                # pc_samp.append(pc_out[idx][samp[i]])
            nsamp += nsel

    nQ = 4*np.ones(nsamp)
    survey_class = 6*np.ones(nsamp)
    vis_class = np.zeros(nsamp)
    post_class = np.zeros(nsamp)
    A_r = np.zeros(nsamp)
    print nsamp, ' galaxies after resampling'

    # Assign surface brightness and fibre mag from fits to observed relations
    # (Loveday+ 2012, App A1)
    mapp = np.array([mapp_out[i] for i in samp_list])
    Mabs = np.array([Mabs_out[i] for i in samp_list])
    sb = 22.42 + 0.029*Mabs + np.random.normal(0.0, 0.76, nsamp)
    imcomp = np.interp(sb, sb_tab, comp_tab)
    r_fibre = 5.84 + 0.747*mapp + np.random.normal(0.0, 0.31, nsamp)
    zcomp = z_comp(r_fibre)
    bad = (imcomp * zcomp < np.random.random(nsamp))
    nQ[bad] = 0
    nbad = len(nQ[bad])
    print nbad, 'out of', nsamp, 'redshifts marked as bad', float(nbad)/nsamp

    # Write out as FITS file
    cols = [fits.Column(name='r_petro', format='E', array=mapp),
            fits.Column(name='fibermag_r', format='E', array=r_fibre),
            fits.Column(name='r_sb', format='E', array=sb),
            fits.Column(name='z_tonry', format='E', 
                          array=[z_out[i] for i in samp_list]),
            fits.Column(name='nQ', format='I', array=nQ),
            fits.Column(name='survey_class', format='I', array=survey_class),
            fits.Column(name='vis_class', format='I', array=vis_class),
            fits.Column(name='post_class', format='I', array=post_class),
            fits.Column(name='ra', format='E', 
                          array=[ra_out[i] for i in samp_list]),
            fits.Column(name='kcorr_r', format='E', 
                          array=[kc_out[i] for i in samp_list]),
            fits.Column(name='pcoeff_r', format='{}E'.format(pdim), 
                          array=[pc_out[i, :] for i in samp_list]),
            fits.Column(name='A_r', format='E', array=A_r)]
    tbhdu = fits.new_table(cols)
    # Need PyFits 3.1 to add new header parameters in this way
    hdr = tbhdu.header
    hdr['omega_l'] = header['omega_l']
    hdr['z0'] = header['z0']
    hdr['area'] = area_dg2
    hdr['alpha'] = alpha
    hdr['Mstar'] = Mstar
    hdr['phistar'] = phistar
    hdr['Q'] = Q
    hdr['P'] = P
    tbhdu.writeto(outfile, clobber=True)

    plt.clf()
    ax = plt.subplot(2, 1, 1)
    ax.plot(zcen, zhist)
    ax.step(zcen, hist_gen, where='mid')
    ax.set_xlabel('z')
    ax.set_ylabel('N(z)')
    ax = plt.subplot(2, 1, 2)
    ax.hist(Mabs, 
            bins=int(4*(Mrange[1] - Mrange[0])), range=(Mrange[0], Mrange[1]))
    ax.set_xlabel('Abs mag M')          
    ax.set_ylabel(r'$N(M)$')    
    plt.draw()
    
#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------ 

class CosmoLookup():
    """Distance and volume-element lookup tables.
    NB volume element is differential per unit solid angle."""

    def __init__(self, H0, omega_l, zlimits, P=1, nz=1000, ev_model='z'):
        cosmo = FlatLambdaCDM(H0=H0, Om0=1-omega_l)
        self._P = P
        self._ev_model = ev_model
        self._H0 = H0
        self._zrange = zlimits
        self._z = np.linspace(zlimits[0], zlimits[1], nz)
        self._dm = cosmo.comoving_distance(self._z).value
        self._dV = cosmo.differential_comoving_volume(self._z).value
        self._dist_mod = cosmo.distmod(self._z).value
        print('CosmoLookup: H0={}, Omega_l={}, P={}'.format(H0, omega_l, P))

    def dm(self, z):
        """Comoving distance."""
        return np.interp(z, self._z, self._dm)

    def dl(self, z):
        """Luminosity distance."""
        return (1+z)*np.interp(z, self._z, self._dm)

    def da(self, z):
        """Angular diameter distance."""
        return np.interp(z, self._z, self._dm)/(1+z)

    def dV(self, z):
        """Volume element per unit solid angle."""
        return np.interp(z, self._z, self._dV)

    def dist_mod(self, z):
        """Distance modulus."""
        return np.interp(z, self._z, self._dist_mod)

    def dist_mod_ke(self, z, kcoeff, kcorr, ecorr):
        """Returns the K- and e-corrected distance modulus
        DM(z) + k(z) - e(z)."""
        dm = self.dist_mod(z) + kcorr(z, kcoeff) - ecorr(z)
        return dm

    def den_evol(self, z):
        """Density evolution at redshift z."""
        if self._ev_model == 'none':
            try:
                return np.ones(len(z))
            except TypeError:
                return 1.0
        if self._ev_model == 'z':
            return 10**(0.4*self._P*z)
        if self._ev_model == 'z1z':
            return 10**(0.4*self._P*z/(1+z))

    def vol_ev(self, z):
        """Volume element multiplied by density evolution."""
        pz = self.dV(z) * self.den_evol(z)
        return pz

    def z_at_dm(self, dm):
        """Redshift at corresponding comoving distance."""
        return np.interp(dm, self._dm, self._z)

#------------------------------------------------------------------------------
# Support functions
#------------------------------------------------------------------------------

def kcorr(z, kcoeff):
    """K-correction from polynomial fit."""
    return np.polynomial.polynomial.polyval(z - par['z0'], kcoeff)

def ran_fun(f, xmin, xmax, nran, args=None, nbin=1000):
    """Generate nran random points according to pdf f(x)"""

    x = np.linspace(xmin, xmax, nbin)
    if args:
        p = f(x, *args)
    else:
        p = f(x)
    return ran_dist(x, p, nran)

def z_comp(r_fibre):
    """Sigmoid function fit to redshift succcess given r_fibre, from misc.zcomp."""
    p = (22.42, 2.55, 2.24)
    return (1.0/(1 + np.exp(p[1]*(r_fibre-p[0]))))**p[2]

def ecorr(z, Q):
    """e-correction."""
    assert par['ev_model'] in ('z', 'z1z')
    if par['ev_model'] == 'z':
        return Q*(z - par['z0'])
    if par['ev_model'] == 'z1z':
        return Q*z/(1+z)
    
def ran_dist(x, p, nran):
    """Generate nran random points according to distribution p(x)"""

    if np.amin(p) < 0:
        print('ran_dist warning: pdf contains negative values!')
    cp = np.cumsum(p)
    y = (cp - cp[0]) / (cp[-1] - cp[0])
    r = np.random.random(nran)
    return np.interp(r, y, x)