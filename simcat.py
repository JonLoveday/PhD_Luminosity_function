#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import math
import mpmath
import numpy as np
import astropy.io.fits as fits
#import pyqt_fit.kde
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import scipy.stats
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from astropy import units as u
from tqdm.notebook import tqdm
import pandas as pd
from kcorrect.kcorrect import Kcorrect
from PhD_Luminosity_function_final import *

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

def simcat_GII(nsim=10):
    """Generate nsim simulated catalogues corresponding to GAMA-II."""

    hdul = fits.open('TilingCatv46.fits')
    data = hdul[1].data
    t=Table(data)
    df = t.to_pandas()

    hdul = fits.open('ApMatchedCatv06.fits')
    data = hdul[1].data
    t=Table(data)
    df2 = t.to_pandas()

    hdul = fits.open('GalacticExtinctionv03.fits')
    data = hdul[1].data
    t=Table(data)
    df3 = t.to_pandas()

    df = pd.merge(df, df2[['CATAID', 'FLUX_AUTO_u', 'FLUX_AUTO_g', 'FLUX_AUTO_r', 'FLUX_AUTO_i', 'FLUX_AUTO_z', 'FLUXERR_AUTO_u', 'FLUXERR_AUTO_g', 'FLUXERR_AUTO_r', 'FLUXERR_AUTO_i', 'FLUXERR_AUTO_z']], on='CATAID', how='left')
    df = pd.merge(df, df3[['CATAID', 'A_u', 'A_g', 'A_r', 'A_i', 'A_z']], on='CATAID', how='left')
    df = df[(df['SURVEY_CLASS']>=4) & (df['NQ']>=3) & (df['Z']>0.002) & (df['Z']<0.65)]

    df['FLUX_AUTO_u'] = df['FLUX_AUTO_u'] * 10**(0.4 * df['A_u'])
    df['FLUX_AUTO_g'] = df['FLUX_AUTO_g'] * 10**(0.4 * df['A_g'])
    df['FLUX_AUTO_r'] = df['FLUX_AUTO_r'] * 10**(0.4 * df['A_r'])
    df['FLUX_AUTO_i'] = df['FLUX_AUTO_i'] * 10**(0.4 * df['A_i'])
    df['FLUX_AUTO_z'] = df['FLUX_AUTO_z'] * 10**(0.4 * df['A_z'])

    df = add_column(df, column_name='Z_TONRY')
    df = kcorrection(
        df, responses = ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'],
        fnames = ['FLUX_AUTO_u', 'FLUX_AUTO_g', 'FLUX_AUTO_r', 'FLUX_AUTO_i', 'FLUX_AUTO_z'],
        ferrnames = ['FLUXERR_AUTO_u', 'FLUXERR_AUTO_g', 'FLUXERR_AUTO_r', 'FLUXERR_AUTO_i', 'FLUXERR_AUTO_z'],
        rband = 'FLUX_AUTO_r', zband = 'FLUX_AUTO_z', redshift = 'Z_TONRY',
        survey='GAMAII')

    df = luminosity_distance(df, redshift='Z_TONRY')
    df = magnitude(df, bands = ['u', 'g', 'r', 'i', 'z'],
                   fluxbands = ['FLUX_AUTO_u', 'FLUX_AUTO_g', 'FLUX_AUTO_r', 'FLUX_AUTO_i', 'FLUX_AUTO_z'],
                   lumdist = 'Lum_Distance', kcorrection = 'Kcorrection')
    for isim in range(nsim):
        outfile=f'jswml_adrien/GII_sim_{isim}.pkl'
        simcat(df, outfile)

def simcat(infile, outfile='jswml_adrien/GII_sim.pkl',
           alpha=-1.23, Mstar=-20.70, phistar=0.01, Q=0.7, P=1.8, chi2max=10, 
           Mrange=(-24, -12), mrange=(10, 19.8), zrange=(0.002, 0.65), nz=65, 
           fbad=0.03, do_kcorr=True, area_fac=1.0, nblock=500000, schec_nz=0, 
           survey='GAMAII', area=180, apply_incomp=True,
           kc_responses=['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'],
           r_band_index=2, p = (22.42, 2.55, 2.24),
           rdenfile='RadialDensityv01.fits'):
    """Generate test data for jswml - see Cole (2011) Sec 5."""

    def gam_dv(z):
        """Gamma function times volume element to integrate."""
        M1 = mrange[1] - cosmo.dist_mod(z) - kcorr(z, c_med) + Q*(z-par['z0'])
        M1 = max(min(Mrange[1], M1), Mrange[0])
        M2 = mrange[0] - cosmo.dist_mod(z) - kcorr(z, c_med) + Q*(z-par['z0'])
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
    tbdata = infile
    H0 = 100.0
    omega_l = 0.7
    par['z0'] = 0
    area_dg2 = area_fac*area
    area = area_dg2*(math.pi/180.0)*(math.pi/180.0)
    cosmo = CosmoLookup(H0, omega_l, zrange)
#    rmin, rmax = cosmo.dm(zrange[0]), cosmo.dm(zrange[1])
    kc = Kcorrect(responses=kc_responses)
    par['kc'] = kc
    par['r_index'] = r_band_index

    if do_kcorr:
        if survey=='GAMAII':
            sel = ((tbdata['SURVEY_CLASS'] >= 4) * 
                   (tbdata['Z_TONRY'] >= zrange[0]) * 
                   (tbdata['Z_TONRY'] <= zrange[1]) * 
                   (tbdata['NQ'] >= 3))
            for ic in range(5):
                sel *= np.isfinite([x[ic] for x in tbdata['coeffs']])
            tbdata = tbdata[sel]
            tbdata.reset_index(drop=True, inplace=True)
            nk = len(tbdata)
            ra_gal = tbdata['RA']
            pc = np.array(tbdata['pcoeffs'].tolist())
            c = np.array(tbdata['coeffs'].tolist())
            pdim = (c.shape)[1]
        elif survey=='GAMAIII':
            sel = ((tbdata['SC'] >= 7) * 
                   (tbdata['Z'] >= zrange[0]) * 
                   (tbdata['Z'] <= zrange[1]) * 
                   (tbdata['NQ'] >= 3))
            for ic in range(5):
                sel *= np.isfinite([x[ic] for x in tbdata['coeffs']])
            tbdata = tbdata[sel]
            tbdata.reset_index(drop=True, inplace=True)
            nk = len(tbdata)
            ra_gal = tbdata['RAcen']
            pc = np.array(tbdata['pcoeffs'].tolist())
            c = np.array(tbdata['coeffs'].tolist())
            pdim = (c.shape)[1]
        else:
            raise ValueError('Survey not supported') 
        
        # For median k-correction, find median of K(z) and fit poly to this,
        # rather than taking median of coefficients
        zbin = np.linspace(zrange[0], zrange[1], 50)
#         k_array = np.polynomial.polynomial.polyval(zbin, pc)
#         kmin = np.min(k_array)
#         kmax = np.max(k_array)
#         print('kmin, kmax =', kmin, kmax)
# #        pdb.set_trace()
#         k_median = np.median(k_array, axis=0)
#         pc_med = np.polynomial.polynomial.polyfit(zbin, k_median, pdim-1)
#         k_fit = np.polynomial.polynomial.polyval(zbin, pc_med)
        c_med = np.median(tbdata['coeffs'], axis=0)
        k_fit = kc.kcorrect(redshift=zbin, coeffs=np.broadcast_to(c_med, (len(zbin), len(c_med))), band_shift = par['z0'])
        kmin = np.min(tbdata['r_Kcorrection'])
        kmax = np.max(tbdata['r_Kcorrection']) 
        print('kmin, kmax =', kmin, kmax)
        
        plt.clf()
#         plt.plot(zbin + par['z0'], k_median)
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
        c = np.zeros((nk, pdim))
        c_med = np.zeros(pdim)

    # zbins = np.linspace(zrange[0], zrange[1], nz+1)
    # V_int = area/3.0 * cosmo.dm(zbins)**3
    # V = np.diff(V_int)
    # zcen = zbins[:-1] + 0.5 * (zbins[1]-zbins[0])
    # hist_gen = np.zeros(nz)

    # Predicted N(z) from integrating evolving Schechter function
    if schec_nz:
        zlims = np.linspace(0.0, 0.65, 66)
        nz = len(zlims) - 1
        with open(schec_nz, 'w') as fout:
            print('alpha {}, M* {}, phi* {}, Q {}, P {}'.format(
                alpha, Mstar, phistar, Q, P), file=fout)
            for i in range(nz):
                zlo = zlims[i]
                zhi = zlims[i+1]
                schec_num, err = scipy.integrate.quad(
                    gam_dv, zlo, zhi, epsabs=1e-3, epsrel=1e-3)
                print(0.5*(zlo+zhi), schec_num, file=fout)
        return

    # Integrate evolving LF for number of simulated galaxies
    nsim, err = scipy.integrate.quad(gam_dv, zrange[0], zrange[1],
                                     epsabs=1e-3, epsrel=1e-3)
    nsim = int(nsim)
    print('Simulating', nsim, 'galaxies')
    mapp_out = np.zeros(nsim)
    Mabs_out = np.zeros(nsim)
    z_out = np.zeros(nsim)
    ra_out = np.zeros(nsim)
    kc_out = np.zeros(nsim)
    pc_out = np.zeros((nsim, 5))
    c_out = np.zeros((nsim, 5))
    
    nrem = nsim
    nout = 0
    while nrem > 0:
        # z, Mabs = util.ran_fun2(zM_pdf, zrange[0], zrange[1], 
        #                         Mrange[0], Mrange[1], nblock)
        z = ran_fun(vol_ev, zrange[0], zrange[1], nblock)
        Mabs = ran_fun(schec, Mrange[0], Mrange[1], nblock) - Q*(z-par['z0'])

        # Calculate apparent mag and test for visibility
        # First do a crude cut without k-corrections
        mapp = Mabs + cosmo.dist_mod(z)
        sel = (mapp >= mrange[0] - kmax) * (mapp < mrange[1] - kmin)
        z, Mabs, mapp = z[sel], Mabs[sel], mapp[sel]
        nsel = len(z)

        # K-corrections for remaining objects
        kidx = np.random.randint(0, nk, nsel)
        pc_ran = pc[kidx, :]
        c_ran = c[kidx, :]
        kc = np.array([kcorr(z[i], c_ran[i, :]) for i in range(nsel)])
        mapp += kc
        ra = ra_gal[kidx]
        sel = (mapp >= mrange[0]) * (mapp < mrange[1])
        z, Mabs, mapp, ra, kc, pc_ran, c_ran = z[sel], Mabs[sel], mapp[sel], ra[sel], kc[sel], pc_ran[sel, :], c_ran[sel, :]
        nsel = len(z)
        if nsel > nrem:
            nsel = nrem
            z, Mabs, mapp, ra, kc, pc_ran, c_ran = z[:nrem], Mabs[:nrem], mapp[:nrem], ra[:nrem], kc[:nrem], pc_ran[:nrem, :], c_ran[:nrem, :]

        mapp_out[nout:nout+nsel] = mapp
        Mabs_out[nout:nout+nsel] = Mabs
        z_out[nout:nout+nsel] = z
        ra_out[nout:nout+nsel] = ra
        kc_out[nout:nout+nsel] = kc
        pc_out[nout:nout+nsel, :] = pc_ran
        c_out[nout:nout+nsel, :] = c_ran

        nout += nsel
        nrem -= nsel
        print(nrem)

    if rdenfile:
        # Apply density fluctuations observed in GAMA data
        rden = Table.read(rdenfile)
        zcen = rden['Z']
        nz = len(zcen)
        dz = zcen[1] - zcen[0]
        zbins = zcen - 0.5*dz
        zbins = np.hstack((zbins, zcen[-1] + dz))
        delta = rden['delta_AV'] - 1
    else:
        # Randomly resample in redshift bins to induce density fluctuations
        zbins = np.linspace(zrange[0], zrange[1], nz+1)
        V_int = area/3.0 * cosmo.dm(zbins)**3
        V = np.diff(V_int)
        zcen = zbins[:-1] + 0.5 * (zbins[1]-zbins[0])
        delta = np.random.normal(0.0, math.sqrt(J3/(V/(u.Mpc**3))))

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
    print('iz  delta  zhist  nsel')
    for iz in range(nz):
        zlo = zbins[iz]
        zhi = zbins[iz+1]
        nsel = int(round((1+delta[iz]) * zhist[iz]))
        hist_gen[iz] = nsel
        print(iz, delta[iz], zhist[iz], nsel)
        if nsel > 0:
            sel = (zlo <= z_out) * (z_out < zhi)
            idx = np.where(sel)
            samp = np.random.randint(0, zhist[iz], nsel)
            for i in range(nsel):
                samp_list.append(idx[0][samp[i]])
                # mapp_samp.append(mapp_out[idx][samp[i]])
                # Mabs_samp.append(Mabs_out[idx][samp[i]])
                # z_samp.append(z_out[idx][samp[i]])
                # ra_samp.append(ra_out[idx][samp[i]])
                # kc_samp.append(kc_out[idx][samp[i]])
                # pc_samp.append(pc_out[idx][samp[i]])
            nsamp += nsel

    nQ = 4*np.ones(nsamp)
    if survey=='GAMAIII':
        survey_class = 7*np.ones(nsamp)
    else:
        survey_class = 6*np.ones(nsamp)
    vis_class = np.zeros(nsamp)
    post_class = np.zeros(nsamp)
    A_r = np.zeros(nsamp)
    print(nsamp, ' galaxies after resampling')

    if apply_incomp:
        # Assign surface brightness and fibre mag from fits to observed relations
        # (Loveday+ 2012, App A1)
        mapp = np.array([mapp_out[i] for i in samp_list])
        Mabs = np.array([Mabs_out[i] for i in samp_list])
        sb = 22.42 + 0.029*Mabs + np.random.normal(0.0, 0.76, nsamp)
        imcomp = np.interp(sb, sb_tab, comp_tab)
        r_fibre = 5.84 + 0.747*mapp + np.random.normal(0.0, 0.31, nsamp)
        zcomp = z_comp(r_fibre, p)
        bad = (imcomp * zcomp < np.random.random(nsamp))
        nQ[bad] = 0
        nbad = len(nQ[bad])
        print(nbad, 'out of', nsamp, 'redshifts marked as bad', float(nbad)/nsamp)
    
    if survey=='GAMAII':
        data = {
            'R_PETRO': mapp, 
            'FIBERMAG_R': r_fibre, 
            'R_SB': sb, 
            'Z_TONRY': [z_out[i] for i in samp_list], 
            'NQ': nQ, 
            'SURVEY_CLASS': survey_class, 
            'vis_class': vis_class, 
            'post_class': post_class, 
            'RA': [ra_out[i] for i in samp_list], 
            'r_Kcorrection': [kc_out[i].tolist() for i in samp_list], 
            'pcoeffs': [pc_out[i].tolist() for i in samp_list], 
            'coeffs': [c_out[i].tolist() for i in samp_list], 
            'A_r': A_r
        }
    elif survey=='GAMAIII':
        data = {
            'm_r': mapp, 
            'FIBERMAG_R': r_fibre, 
            'R_SB': sb, 
            'Z': [z_out[i] for i in samp_list], 
            'NQ': nQ, 
            'SC': survey_class, 
            'vis_class': vis_class, 
            'post_class': post_class, 
            'RAcen': [ra_out[i] for i in samp_list], 
            'r_Kcorrection': [kc_out[i].tolist() for i in samp_list], 
            'pcoeffs': [pc_out[i].tolist() for i in samp_list], 
            'coeffs': [c_out[i].tolist() for i in samp_list], 
            'A_r': A_r
        }
    else:
        raise ValueError('Survey not supported')

    # Create the DataFrame
    df = pd.DataFrame(data)
    df.to_pickle(outfile)

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

    return df    

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
        self._dm = cosmo.comoving_distance(self._z)
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

    def dist_mod_ke(self, z, coeff, kcorr, ecorr):
        """Returns the K- and e-corrected distance modulus
        DM(z) + k(z) - e(z)."""
        dm = self.dist_mod(z) + kcorr(z, coeff) - ecorr(z)
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

def kcorr(z, coeffs):
    """K-correction from polynomial fit."""
#     return np.polynomial.polynomial.polyval(z - par['z0'], kcoeff)
    
    if isinstance(z, float):
        z = np.array([z])

    if np.ndim(coeffs) == 1:
        coeffs = [coeffs]    
        kcorrect = np.zeros((len(coeffs), len(z)))
        for i in range(len(coeffs)):
            kcorrect[i] = [x[par['r_index']] for x in par['kc'].kcorrect(redshift=z, 
                                                                  coeffs=np.broadcast_to(coeffs[i], (len(z), len(coeffs[i]))), 
                                                                  band_shift = par['z0'])]
            kcorrect = kcorrect[0][0]
    else :
        kcorrect = np.zeros((len(coeffs), len(z)))
        for i in range(len(coeffs)):
            kcorrect[i] = [x[par['r_index']] for x in par['kc'].kcorrect(redshift=z, 
                                                                  coeffs=np.broadcast_to(coeffs[i], (len(z), len(coeffs[i]))), 
                                                                  band_shift = par['z0'])]        
    return kcorrect

def ran_fun(f, xmin, xmax, nran, args=None, nbin=1000):
    """Generate nran random points according to pdf f(x)"""

    x = np.linspace(xmin, xmax, nbin)
    if args:
        p = f(x, *args)
    else:
        p = f(x)
    return ran_dist(x, p, nran)

def z_comp(r_fibre, p):
    """Sigmoid function fit to redshift succcess given r_fibre, from misc.zcomp."""
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