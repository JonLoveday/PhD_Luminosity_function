#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
#import pmap
import pickle
import astropy.io.fits as fits
#import pyqt_fit.kde
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import scipy.stats
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm
import pandas as pd
from kcorrect.kcorrect import Kcorrect

#------------------------------------------------------------------------------
# Parameters
#------------------------------------------------------------------------------

# Avoid excessive space around markers in legend
matplotlib.rcParams['legend.handlelength'] = 0

# Treatment of numpy errors
np.seterr(all='warn')

# Global parameters
par = {'progName': 'jswml.py', 'version': 1.1, 'ev_model': 'z',
       'clean_photom': True, 'use_wt': True, 'kc_use_poly': False}
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

def ev_fit_sim(isim):
    infile = pd.read_pickle(f'jswml_adrien/GII_sim_{isim}.pkl')
    outfile = f'jswml_adrien/jswml_GII_sim_{isim}.pkl'
    ev_fit(infile, outfile)


def ev_fit_sim_III(isim):
    infile = pd.read_pickle(f'jswml_adrien/GIII_sim_{isim}.pkl')
    outfile = f'jswml_adrien/jswml_GIII_sim_{isim}.pkl'
    ev_fit(infile, outfile)


def ev_fit_sim_post(isim):
    infile = pd.read_pickle(f'jswml_adrien/GII_sim_{isim}.pkl')
    outfile = f'jswml_adrien/jswml_post_GII_sim_{isim}.pkl'
    ev_fit(infile, outfile, method='post')

def ev_fit_sim_post_III(isim):
    infile = pd.read_pickle(f'jswml_adrien/GIII_sim_{isim}.pkl')
    outfile = f'jswml_adrien/jswml_post_GIII_sim_{isim}.pkl'
    ev_fit(infile, outfile, method='post')


def ev_fit(infile, outfile, mlims=(0, 19.8), param='R_PETRO',
           Mmin=-24, Mmax=-12, Mbin=48, dmlim=2,
           zmin=0.002, zmax=0.65, nz=65,
           lf_zbins=((0, 20), (20, 65)),
           Pbins=(-0.5, 4.0, 45), Qbins=(0.0, 1.5, 30),
           P_prior=(2, 1), Q_prior=(1, 1),
           idebug=1, method='lfchi', err_type='jack', use_mp=False, opt=True,
           lf_est='weight', survey='GAMAII', area=180, kc_responses=['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0'], r_band_index=2, p = (22.42, 2.55, 2.24)):
    """Fit evolution parameters and radial overdensities.
    Searches over both Q and P values,
    rather than trying to estimate P from Cole eqn (25).

    Elements of P_prior and Q_prior are mean and variance."""

    kc = Kcorrect(responses=kc_responses)    
    
    global par
    par.update({'infile': infile, 'mlims': mlims,
                'param': param, 'dmlim': dmlim,
                'zmin': zmin, 'zmax': zmax,
                'Mmin': Mmin, 'Mmax': Mmax, 'Mbin': Mbin,
                'idebug': idebug, 'method': method, 'lf_est': lf_est, 'survey': survey, 'area': area, 'kc': kc, 'r_index': r_band_index, 'p': p})

    print('\n************************\njswml.py version ', par['version'])
    print('survey : ', survey)
    print('mlims : ', mlims)
    print('redshift range : [', zmin, '; ', zmax, ']') 
    print('area : ', area)
    print('method : ', method)
    print('Kcorrect responses : ', kc_responses)
    print(sel_dict)
    assert method in methods
    lf_bins = np.linspace(Mmin, Mmax, Mbin+1)

    samp = Sample(infile, par, sel_dict)
    costfn = Cost(samp, nz, (zmin, zmax), lf_bins, lf_zbins, method,
                  P_prior, Q_prior, Qbins[0], Qbins[1], err_type)
    out = {'par': par}

    if par['idebug'] > 0:
        print('Q, P chi^2 grid using', method)

    # Calculate chi^2 on (P,Q) grid to get likelihood contours and to find
    # starting point for minimization
    Qmin = Qbins[0]
    Qmax = Qbins[1]
    nQ = Qbins[2]
    Qstep = float(Qmax - Qmin)/nQ
    Pmin = Pbins[0]
    Pmax = Pbins[1]
    nP = Pbins[2]
    Pstep = float(Pmax - Pmin)/nP
    Qa = np.linspace(Qmin, Qmax, nQ, endpoint=False) + 0.5*Qstep
    Pa = np.linspace(Pmin, Pmax, nP, endpoint=False) + 0.5*Pstep
    plt.clf()

    if use_mp:
        chi2grid = np.array(pmap.parallel_map(lambda Q: [costfn((P, Q)) 
                                                         for P in Pa], Qa))
    else:
        chi2grid = np.array([[costfn((P, Q)) for P in Pa] for Q in tqdm(Qa)])

    
    extent = (Pmin, Pmax, Qmin, Qmax)
    cmap = matplotlib.cm.jet
    ax = plt.subplot(313)
    im = ax.imshow(chi2grid, cmap=cmap, aspect='auto', origin='lower', 
                   extent=extent, interpolation='nearest')
    cb = plt.colorbar(im, ax=ax)
    (j, i) = np.unravel_index(np.argmin(chi2grid), chi2grid.shape)
    P_maxl = Pmin + (i+0.5)*Pstep
    Q_maxl = Qmin + (j+0.5)*Qstep
    ax.plot(P_maxl, Q_maxl, '+')
    ax.set_xlabel('P')
    ax.set_ylabel('Q')
    # Add contour at chi²_min + 4
    chi2_min = chi2grid.min()
    contour_level = chi2_min + 4
    ax.contour(Pa, Qa, chi2grid, levels=[contour_level], colors='white')
    plt.draw()

    if opt:
        if par['idebug'] > 0:
            print('Simplex optimization ...')
        # Use simplex method to optimize ev parameters (P,Q)
        res = scipy.optimize.fmin(costfn, (P_maxl, Q_maxl), xtol=0.1, ftol=0.1, 
                                  full_output=True)
        Popt = res[0][0]
        Qopt = res[0][1]
    else:
        c = costfn((P_maxl, Q_maxl))
        Popt = P_maxl
        Qopt = Q_maxl

    out['Pbins'] = Pbins
    out['Qbins'] = Qbins
    out['chi2grid'] = chi2grid
    out['Pa'] = Pa
    out['Qa'] = Qa
    out['P'] = Popt
    out['P_err'] = 0
    out['Q'] = Qopt
    out['Q_err'] = 0
    out['zbin'] = costfn.zbin
    out['delta'] = costfn.delta
    out['delta_err'] = costfn.delta_err
    out['den_var'] = costfn.den_var
    out['lf_bins'] = lf_bins
    out['phi'] = costfn.phi
    out['phi_err'] = costfn.phi_err
    out['Mbin'] = costfn.Mbin
    out['Mhist'] = costfn.Mhist
    out['whist'] = costfn.whist
    out['ev_fit_chisq'] = costfn.chisq
    out['ev_fit_nu'] = costfn.nu
    fout = open(outfile, 'wb')
    pickle.dump(out, fout)
    fout.close()
    
#------------------------------------------------------------------------------
# Classes
#------------------------------------------------------------------------------  

class Sample(object):
    """A sample of galaxies, whose attributes are stored in
    structured array gal_arr."""

    def __init__(self, infile, selpar, sel_dict, Q=0, chi2max=10, nqmin=3):
        """Read selected objects from FITS table."""

        global cosmo, par

        par = selpar
        zmin, zmax = par['zmin'], par['zmax']
        self.Mmin = par['Mmin']
        self.Mmax = par['Mmax']
        self.Mbin = par['Mbin']
        self.Mstep = float(self.Mmax - self.Mmin)/self.Mbin

#         hdulist = fits.open(infile)
#         header = hdulist[1].header
#         tbdata = hdulist[1].data
#         cols = hdulist[1].columns
        tbdata = infile
        par['H0'] = 100.0
        par['omega_l'] = 0.7
        par['z0'] = 0
        par['area'] = par['area'] * (math.pi/180.0)**2
        cosmo = CosmoLookup(par['H0'], par['omega_l'], (zmin, zmax))
        self.par = par
        self.cosmo = cosmo
        print('H0, omega_l, z0, area/Sr = ',
              par['H0'], par['omega_l'], par['z0'], par['area'])

        try:
            alpha = header['alpha']
            sim = True
            print('Simulated data')
        except:
            sim = False

        mock = False
        if par['survey'] == 'GAMAII':
            sel = ((tbdata['SURVEY_CLASS'] >= 4) * 
                   (tbdata['NQ'] >= nqmin) * 
                   (tbdata['Z_TONRY'] >= zmin) * 
                   (tbdata['Z_TONRY'] <= zmax))
        elif par['survey'] == 'GAMAIII':
            sel = ((tbdata['SC'] >= 7) * 
                   (tbdata['NQ'] >= nqmin) * 
                   (tbdata['Z'] >= zmin) * 
                   (tbdata['Z'] <= zmax))
        else:
            raise ValueError('Survey not supported')

            # Apply other selection limits in sel_dict
        for key, limits in sel_dict.items():
            print(key, limits)
            sel *= ((tbdata[key] >= limits[0]) *
                    (tbdata[key] < limits[1]))
            par[key] = limits

#         # Exclude objects with suspect photometry
#         # pdb.set_trace()
#         if par['clean_photom']:
#             ncand = len(tbdata[sel])
#             sel *= ((tbdata['bn_objid'] < 0) *
#                     (np.fabs(tbdata['r_petro'] - tbdata['r_sersic']) <
#                     par['dmlim']))
#             nclean = len(tbdata[sel])
#             print(nclean, 'out of', ncand, 'targets with clean photometry')

        tbdata = tbdata[sel]
        tbdata.reset_index(drop=True, inplace=True)
        ngal = len(tbdata)
        pcoeffs_array = np.array(tbdata['pcoeffs'].tolist())
        nk = pcoeffs_array.shape[1]
        gal_arr = np.zeros(
            ngal,
            dtype=[('cataid', 'int64'),
                   ('appval_sel', 'float32'), ('absval_sel', 'float32'),
                   ('appval_lf', 'float32'), ('absval_lf', 'float32'),
                   ('ra', 'float32'), ('dec', 'float32'),
                   ('weight', 'float32'),
                   ('kc', 'float32'), ('kcoeff', 'float32', nk), ('coeff', 'float32', nk),
                   ('z', 'float32'), ('zlo', 'float32'), ('zhi', 'float32')
                   ])
        
        if par['survey'] == 'GAMAII':  
            z = tbdata['Z_TONRY']
            gal_arr['ra'] = tbdata['RA']
            gal_arr['appval_sel'] = tbdata['R_PETRO']
            gal_arr['appval_lf'] = tbdata[par['param']]
            gal_arr['z'] = z
            gal_arr['kc'] = tbdata['r_Kcorrection']
            gal_arr['kcoeff'] = np.array(tbdata['pcoeffs'].tolist())
            gal_arr['coeff'] = np.array(tbdata['coeffs'].tolist())        
        elif par['survey'] == 'GAMAIII':  
            z = tbdata['Z']
            gal_arr['ra'] = tbdata['RAcen']
            gal_arr['appval_sel'] = tbdata['m_r']
            gal_arr['appval_lf'] = tbdata[par['param']]
            gal_arr['z'] = z
            gal_arr['kc'] = tbdata['r_Kcorrection']
            gal_arr['kcoeff'] = np.array(tbdata['pcoeffs'].tolist())
            gal_arr['coeff'] = np.array(tbdata['coeffs'].tolist())
        else:
            raise ValueError('Survey not supported')         
            
        if par['kc_use_poly']:
            gal_arr['kc'] = np.polynomial.polynomial.polyval(
                z - par['z0'], gal_arr['kcoeff'].transpose(), tensor=False)
        if sim:
            # Reverse coeffs given in old (high -> low) order
            gal_arr['kcoeff'] = gal_arr['kcoeff'][:, ::-1]

#         # Fit polynomial to median K(z) for good fits
#         good = np.isfinite(gal_arr['kc']) * (tbdata['chi2'] < chi2max)
#         zbin = np.linspace(par['zmin'], par['zmax'], 50) - par['z0']
#         k_array = np.polynomial.polynomial.polyval(
#             zbin, gal_arr['kcoeff'][good].transpose())
#         k_median = np.median(k_array, axis=0)
#         self.kmean = np.polynomial.polynomial.polyfit(zbin, k_median, nk-1)
        self.kmean = [np.median([x[0] for x in gal_arr['coeff']]), np.median([x[1] for x in gal_arr['coeff']]), np.median([x[2] for x in gal_arr['coeff']]), np.median([x[3] for x in gal_arr['coeff']]), np.median([x[4] for x in gal_arr['coeff']])]

#         # Set any missing or bad k-corrs to median values
#         bad = np.logical_not(good)
#         nbad = len(z[bad])
#         if nbad > 0:
#             gal_arr['kc'][bad] = np.polynomial.polynomial.polyval(
#                 z[bad] - par['z0'], self.kmean)
#             gal_arr['kcoeff'][bad] = self.kmean
#             print(nbad, 'missing/bad k-corrections replaced with mean')
#             f = open('bad_kcorr.txt', 'w')
#             for ibad in range(nbad):
#                 print(gal_arr['cataid'][bad][ibad], file=f)
#             f.close()

        gal_arr['absval_sel'] = (gal_arr['appval_sel'] - cosmo.dist_mod(z) -
                                 gal_arr['kc'] + ecorr(z, Q))
        gal_arr['absval_lf'] = (gal_arr['appval_lf'] - cosmo.dist_mod(z) -
                                gal_arr['kc'] + ecorr(z, Q))

#         self.header = header
        self.tbdata = tbdata
        self.gal_arr = gal_arr
        self.ngal = ngal
        print(ngal, 'galaxies selected')

        # Completeness weight
        # sb = (tbdata[('r_petro') + 2.5*lg2pi +
        #                   5*np.log10(tbdata[('petror50_r')))
        try:
            sb = tbdata['R_SB']
            imcomp = np.interp(sb, sb_tab, comp_tab)
        except:
            print('No column r_sb; ignoring SB completeness')
            imcomp = np.ones(ngal)
        try:
            r_fibre = tbdata['FIBERMAG_R']
            zcomp = z_comp(r_fibre)
        except:
            print('No column fibermag_r; ignoring redshift completeness')
            zcomp = np.ones(ngal)
        self.gal_arr['weight'] = np.clip(1.0/(imcomp*zcomp), 1, wmax)

        # Read Vmax values if present
        try:
            self.Vmax_raw = tbdata['Vmax_raw']
            self.Vmax_dc = tbdata['Vmax_dc']
            self.Vmax_dec = tbdata['Vmax_dec']
        except:
            pass

    def calc_limits(self, Q, vis=True):
        """Calculate absolute values and visibilty limits for each galaxy,
        returning a view of gal_arr for galaxies within absolute limits."""

        zmin, zmax = par['zmin'], par['zmax']
        ngal = self.ngal
        z = self.gal_arr['z']
        sel = (z > 0)

        kc = self.gal_arr['kc']
        kcoeff = self.gal_arr['kcoeff']
        coeff = self.gal_arr['coeff']
        self.gal_arr['absval_sel'] = (self.gal_arr['appval_sel'] -
                                      cosmo.dist_mod(z) - kc + ecorr(z, Q))
        self.gal_arr['absval_lf'] = (self.gal_arr['appval_lf'] -
                                     cosmo.dist_mod(z) - kc + ecorr(z, Q))
        if vis:
            self.gal_arr['zlo'] = [zdm(par['mlims'][0] - self.gal_arr['absval_sel'][i], 
                                       coeff[i], (zmin, zmax), Q) for i in range(ngal)]
            self.gal_arr['zhi'] = [zdm(par['mlims'][1] - self.gal_arr['absval_sel'][i], 
                                       coeff[i], (zmin, zmax), Q) for i in range(ngal)]
        # Galaxies within absolute limits
        absm = self.gal_arr['absval_lf']
        sel *= (par['Mmin'] <= absm) * (absm < par['Mmax'])
        
        gala = self.gal_arr[sel]
        if par['idebug'] > 1:
            print(len(gala), 'galaxies satisfy absolute limits')
        return gala

    def abs_bin(self, absval):
        """Returns bin number and fraction for given absval, such that:
        absval = absMin + (iabs+frac)*absStep."""
        
        absval = np.clip(absval, self.Mmin, self.Mmax)
        iabs = np.floor((absval - self.Mmin)/self.Mstep).astype(np.int32)
        iabs = np.clip(iabs, 0, self.Mbin - 1)
        frac = (absval - (self.Mmin + iabs*self.Mstep))/self.Mstep
        return iabs, frac

    def subset(self, idx):
        """Return subset of gala with given indices."""
        subset = Sample(self.type, self.qty_list)
        subset.gal_arr = self.gal_arr[idx]
        subset.ngal = len(subset.gal_arr)
        return subset

    def resample(self):
        """Bootstrap resampling"""
        idx = np.random.randint(0, self.ngal, self.ngal)
        return self.subset(idx)
    
    def jacknife(self, jack):
        """Return a subsample with jacknife region jack omitted"""

        idx = (self.gal_arr['ra'] < ra_jack[jack]) + (self.gal_arr['ra'] >= ra_jack[jack] + 4.0)
        subset = self.subset(idx)
        subset.area *= 8.0/9.0
        return subset

class Cost(object):
    """Cost function and associated parameters."""

    def __init__(self, samp, nz, zminmax, lf_bins, lf_zbins, 
                 method, P_prior, Q_prior, Qmin, Qmax, err_type='jack'):
        (zmin, zmax) = zminmax
        self.samp = samp
        self.nz = nz
        self.zmin = zmin
        self.zmax = zmax
        self.zbin_edges, self.zstep = np.linspace(zmin, zmax, nz+1, retstep=True)
        self.zbin = self.zbin_edges[:-1] + 0.5 * self.zstep
        self.dist_mod = cosmo.dist_mod(self.zbin)
        self.V_int = par['area'] / 3.0 * cosmo.dm(self.zbin_edges)**3
        self.V = np.diff(self.V_int)
        self.lf_bins = lf_bins
        self.lf_zbins = lf_zbins
        self.method = method
        self.P_prior = P_prior
        self.Q_prior = Q_prior
        self.Qmin = Qmin
        self.Qmax = Qmax
        self.err_type = err_type
        self.Q = -99.0
        self.delta_old = np.ones(nz)

        # Mag bin limits for LF
        if par['idebug'] > 0:
            print('Setting LF bin limits Qmin, Qmax = ', Qmin, Qmax)
        if self.method == 'post':
            self.binidx = np.ones(len(self.lf_bins) - 1, dtype=bool)
        if self.method == 'lfchi':
            self.binidx = np.ones((len(self.lf_zbins), len(self.lf_bins) - 1), 
                                  dtype=bool)
            zstep = (zmax - zmin)/nz
            if par['idebug'] > 0:
                print('zlo, zhi, Mmin, Mmax, nbins')

        if self.method == 'post':
            for Q in ((Qmin, Qmax)):
                gala = samp.calc_limits(Q)
                Mhist, edges = np.histogram(gala['absval_lf'], lf_bins)
                if Q == self.Qmin:
                    self.binidx *= (Mhist > 9)
                    print('LF bins at Qmin: ', Mhist)
                if Q == self.Qmax:
                    self.binidx *= (Mhist > 9)
                    print('LF bins at Qmax: ', Mhist)
        
            print(self.binidx)
            
            def process_boolean_array(arr):
                false_indices = [i for i, val in enumerate(arr) if not val]
                if not false_indices:
                    return arr

                mid_index = len(arr) // 2
                for i in false_indices:
                    if i < mid_index and arr[i + 1]:
                        arr[:i+1] = [False] * (i+1)
                    elif i > mid_index and arr[i - 1]:
                        arr[i:] = [False] * (len(arr) - i)

                return arr
            
            self.binidx = process_boolean_array(self.binidx)
            print(self.binidx)
            
        if self.method == 'lfchi':
            for Q in ((Qmin, Qmax)):
                gala = samp.calc_limits(Q)
                for iz in range(len(lf_zbins)):
                    zlo = zmin + lf_zbins[iz][0]*zstep
                    zhi = zmin + lf_zbins[iz][1]*zstep
                    idx = (zlo <= gala['z']) * (gala['z'] < zhi)
                    Mhist, edges = np.histogram(gala['absval_lf'][idx], lf_bins)
                    Mmin = par['mlims'][0] - dmodk(zhi, samp.kmean, Q)
                    Mmax = par['mlims'][1] - dmodk(zlo, samp.kmean, Q)
                    Mlo = edges[:-1]
                    Mhi = edges[1:]
                    if Q == self.Qmin:
                        self.binidx[iz, :] *= (Mhi < Mmax) * (Mlo > Mmin) * (Mhist > 9)
                        print(zlo, zhi, Mmin, Mmax, len(Mhist[self.binidx[iz, :]]))
                        print('LF bins at Qmin: ', Mhist)
                    if Q == self.Qmax:
                        self.binidx[iz, :] *= (Mhi < Mmax) * (Mlo > Mmin) * (Mhist > 9)
                        print(zlo, zhi, Mmin, Mmax, len(Mhist[self.binidx[iz, :]]))
                        print('LF bins at Qmax: ', Mhist)
            
            print(self.binidx)
            
            def process_2d_boolean_array(arr):
                result = []
                for row in arr:
                    false_indices = [i for i, val in enumerate(row) if not val]
                    if not false_indices:
                        result.append(row.tolist())  # Convert NumPy array to list
                        continue

                    mid_index = len(row) // 2
                    for i in false_indices:
                        if i < mid_index and (i + 1 < len(row)) and row[i + 1]:
                            row[:i + 1] = [False] * (i + 1)
                        elif i > mid_index and (i - 1 >= 0) and row[i - 1]:
                            row[i:] = [False] * (len(row) - i)
                    result.append(row.tolist())  # Convert NumPy array to list

                return np.array(result)
            
            self.binidx = process_2d_boolean_array(self.binidx)
            print(self.binidx)

    def __call__(self, PQ):
        """Returns cost for evolution parameters (P, Q).  
        If P is None then solve for P."""
        
        (P, Q) = PQ
        if Q != self.Q:
            self.Q = Q
            self.gala = self.samp.calc_limits(Q)
            self.S_obs, self.S_vis = vis_calc(
                self.gala, self.nz, self.zmin, self.zstep, self.V, self.V_int)
            self.zhist, bin_edges = np.histogram(
                self.gala['z'], self.nz, (self.zmin, self.zmax), 
                weights=self.gala['weight'])
            Mhi = (
                par['Mmax'] - self.dist_mod - 
                kcorr(self.zbin, self.gala['coeff']) + 
                ecorr(self.zbin, Q))
            self.hi_bin, self.hi_frac = self.samp.abs_bin(Mhi)
            Mlo = (
                par['Mmin'] - self.dist_mod - 
                kcorr(self.zbin, self.gala['coeff']) + 
                ecorr(self.zbin, Q))
            self.lo_bin, self.lo_frac = self.samp.abs_bin(Mlo)
            # pdb.set_trace()
        if P is None:
            (converged, self.P, self.mu, Npred, self.delta, 
             self.den_var, Pz, Vdc_max) = delta_P_solve(
                Q, self.gala, self.zbin, self.zhist, self.V, self.V_int, 
                self.S_vis, self.P_prior, self.delta_old)
        else:
            self.P = P
            (converged, Npred, self.delta, self.den_var, 
             Pz, Vdc_max, niter) = delta_solve(
                P, Q, self.gala, self.nz, (self.zmin, self.zmax), self.zbin, 
                self.zhist, self.V, self.V_int, self.S_vis)
        self.delta_old = self.delta

        # Jacknife errors on delta
        if self.err_type == 'jack':
            delta_jack = np.zeros((njack, self.nz))
            for jack in range(njack):
                idx = ((self.gala['ra'] < ra_jack[jack]) + 
                       (self.gala['ra'] >= ra_jack[jack] + 4.0))
                zhist, bin_edges = np.histogram(
                    self.gala['z'][idx], self.nz, (self.zmin, self.zmax), 
                    weights=self.gala['weight'][idx])
                if P is None:
                    xx, xx, delta_jack[jack, :], xx, xx, xx = delta_P_solve(
                        P, Q, self.gala[idx], self.zbin, zhist, 
                        self.V, self.V_int, self.S_vis[:, idx], self.delta_old)
                else:
                    xx, xx, delta_jack[jack, :], xx, xx, xx, xx = delta_solve(
                        P, Q, self.gala[idx], self.nz, (self.zmin, self.zmax), 
                        self.zbin, zhist, self.V, self.V_int, 
                        self.S_vis[:, idx])
                self.delta_err = np.sqrt((njack-1) * np.var(delta_jack, axis=0))
            del_var = self.delta_err**2
        else:
            self.delta_err = np.zeros(self.nz)
            del_var = self.den_var

        ax = plt.subplot(311)
        plt.cla()
        ax.step(self.zbin, self.delta, where='mid')
        ax.errorbar(self.zbin, self.delta, self.delta_err, fmt='none')
        ax.bar(self.zbin - 0.5*self.zstep, 2*np.sqrt(self.den_var), 
               width=self.zstep, bottom=self.delta - np.sqrt(self.den_var), 
               alpha=0.1, ec='none')
        ax.plot([self.zmin, self.zmax], [1.0, 1.0], ':')
        ax.set_ylim(0, 5)
        ax.set_xlabel('Redshift z')          
        ax.set_ylabel(r'$\Delta(z)$')
        ax.text(0.1, 0.9, r'$P = {:4.2f},\ Q = {:4.2f}$'.format(P, Q),
                transform = ax.transAxes)

        lf = lf1d(self.gala, Vdc_max, self.lf_bins)
        (self.Mbin, self.Mhist, self.whist, self.phi, self.phi_err) = (
            lf['Mbin'], lf['Mhist'], lf['whist'], lf['phi'], lf['phi_err'])

        ax = plt.subplot(312)
        plt.cla()
        ax.errorbar(lf['Mbin'], lf['phi'], lf['phi_err'])
        ax.set_xlabel(r'$M_r$')
        ax.set_ylabel(r'$\Phi(M_r)$')
        ax.semilogy(base=10, nonpositive='clip')
        ax.set_ylim(1e-6, 0.05)
        plt.subplots_adjust(hspace=0.25)
        plt.draw()
        
        if self.method == 'post':
            idx = (self.delta > 0) * (del_var > 0) 
            densum = np.sum(self.zhist[idx] *
                            np.log((self.V[idx]/(u.Mpc**3))*Pz[idx]*self.delta[idx]))
            phisum = np.sum(self.whist[self.binidx]*np.log(self.phi[self.binidx]))
#             phisum = np.sum(self.whist*np.log(self.phi))

            sum1 = np.zeros((len(self.gala), len(Pz)))
            for igal in range(len(self.gala)):
                for iz in range(len(Pz)):
                    sum1[igal, iz] = (
                        self.lo_frac[igal, iz]*self.phi[self.lo_bin[igal, iz]] + 
                        self.phi[self.lo_bin[igal, iz]+1:self.hi_bin[igal, iz]-1].sum() +
                        self.hi_frac[igal, iz]*self.phi[self.hi_bin[igal, iz]])
            sum2 = np.dot(sum1, (self.V/(u.Mpc**3))*Pz*self.delta)
            xsum = np.sum(self.gala['weight'] * np.log(sum2))

            lnL = ((densum + phisum - xsum)/len(self.gala) -
                   ((self.delta[idx]-1)**2/(2*del_var[idx])).sum() - 
                   (self.P-self.P_prior[0])**2/(2*self.P_prior[1]) - 
                   (self.Q-self.Q_prior[0])**2/(2*self.Q_prior[1]))

            self.chisq = -2*lnL
            self.nu = len(self.gala) - 2
#             if par['idebug'] > 0:
#                 print('{:5.2f} {:5.2f} {:6d} {:e} {:e} {:e} {:e} {:6d} {:b} {:3d}'.format(
#                     self.Q, self.P, len(self.gala), densum, phisum, xsum, lnL, 
#                     len(self.zhist[idx]), converged, niter))
                # pdb.set_trace()
            if math.isnan(self.chisq):
                self.chisq = 1e9
            return self.chisq

        if self.method == 'denchi':
            self.chisq = ((self.delta-1)**2 / del_var).sum()
            self.nu = len(self.delta) - 2
            if par['idebug'] > 0:
                print('{:5.2f} {:5.2f} {:6d} {:e} {:b} {:3d}'.format(
                    self.Q, self.P, len(self.gala), self.chisq, 
                    converged, niter))
            if not(converged):
                self.chisq *= 10
            return self.chisq

        if self.method == 'lfchi':
            zstep = (self.zmax - self.zmin)/self.nz
            phiz = np.zeros((len(self.lf_zbins), len(self.lf_bins) - 1))
            phiz_err = np.zeros((len(self.lf_zbins), len(self.lf_bins) - 1))
            for iz in range(len(self.lf_zbins)):
                izlo = self.lf_zbins[iz][0]
                izhi = self.lf_zbins[iz][1]
                zlo = self.zmin + izlo*zstep
                zhi = self.zmin + izhi*zstep
                galidx = (zlo <= self.gala['z']) * (self.gala['z'] < zhi)
                galz = self.gala[galidx]
                V_max = np.dot(self.delta[izlo:izhi] * 
                               Pz[izlo:izhi] * self.V[izlo:izhi],
                               self.S_vis[izlo:izhi, galidx])
                lfz = lf1d(galz, V_max, self.lf_bins)
                phiz[iz, :] = lfz['phi']
                phiz_err[iz, :] = lfz['phi_err']
                ax.errorbar(lfz['Mbin'][self.binidx[iz, :]], 
                            phiz[iz, self.binidx[iz, :]], 
                            phiz_err[iz, self.binidx[iz, :]])
                plt.draw()

            idx = del_var > 0
            self.chisq = ((self.delta[idx]-1)**2 / del_var[idx]).sum()
            self.nu = len(self.delta[idx]) - 2
            for iz in range(len(self.lf_zbins) - 1):
                for jz in range(iz+1, len(self.lf_zbins)):
                    idx = self.binidx[iz] * self.binidx[jz]
                    self.nu += len(lfz['Mbin'][idx])
                    self.chisq += np.sum((phiz[iz, idx] - phiz[jz, idx])**2 /
                                  (phiz_err[iz, idx]**2 + phiz_err[jz, idx]**2))

            ax.text(0.1, 0.9, r'$\chi^2 = {:6.1f}$'.format(self.chisq),
                    transform = ax.transAxes)
#             if par['idebug'] > 0:
#                 print('{:5.2f} {:5.2f} {:6d} {:e} {:4d} {:b} {:3d}'.format(
#                     self.Q, self.P, len(self.gala), self.chisq, self.nu, 
#                     converged, niter))
                # pdb.set_trace()
            # if not(converged):
            #     self.chisq *= 10
            return self.chisq

        self.chisq = 0
        self.nu = 0
        popt, pcov = scipy.optimize.curve_fit(lambda x, m, c: m*x + c,
                                              self.zbin, self.delta, p0=(0, 1),
                                              sigma = np.sqrt(del_var))
#         if par['idebug'] > 0:
#             print('{:5.2f} {:5.2f} {:6d} {:e} {:b} {:3d}'.format(
#                     self.Q, self.P, len(self.gala), popt[0], converged, niter))

        if par['idebug'] > 1:
            ax = plt.subplot(2, 1, 1)
            ax.plot((0.0, 0.5), np.polynomial.polynomial.polyval(popt, (0.0, 0.5)), '--')
            plt.draw()

        if self.method == 'min_slope':
            return abs(popt[0])
        else:
            if not(converged):
                if Q <= self.Qmin:
                    return 9
                else:
                    return -9
            return popt[0]    

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

def ecorr(z, Q):
    """e-correction."""
    assert par['ev_model'] in ('z', 'z1z')
    if par['ev_model'] == 'z':
        return Q*(z - par['z0'])
    if par['ev_model'] == 'z1z':
        return Q*z/(1+z)

def z_comp(r_fibre):
    """Sigmoid function fit to redshift succcess given r_fibre, from misc.zcomp."""
    p = par['p']
    return (1.0/(1 + np.exp(p[1]*(r_fibre-p[0]))))**p[2]
    
def zdm(dmod, coeff, zRange, Q):
    """Calculate redshift z corresponding to distance modulus dmod, solves
    dmod = m - M = DM(z) + K(z) - Q(z-z0),
    ie. including k-correction and luminosity evolution Q.
    z is constrained to lie in range zRange."""

    if dmodk(zRange[0], coeff, Q) - dmod > 0:
        return zRange[0]
    if dmodk(zRange[1], coeff, Q) - dmod < 0:
        return zRange[1]
    z = scipy.optimize.brentq(lambda z: dmodk(z, coeff, Q) - dmod,
                              zRange[0], zRange[1], xtol=1e-5, rtol=1e-5)
    return z

def dmodk(z, coeff, Q):
    """Returns the K- and e-corrected distance modulus 
    DM(z) + k(z) - e(z)."""
    dm =  cosmo.dist_mod(z) + kcorr(z, coeff) - ecorr(z, Q)
    return dm

def vis_calc(gala, nz, zmin, zstep, V, V_int):
    """Arrays S_obs and S_vis contain volume-weighted fraction of 
    redshift bin iz in which galaxy igal lies and is visible."""

    afac = par['area'] / 3.0
    ngal = len(gala)
    S_obs = np.zeros((nz, ngal))
    S_vis = np.zeros((nz, ngal))

    for igal in range(ngal):
        ilo = min(nz-1, int((gala['zlo'][igal] - zmin) / zstep))
        ihi = min(nz-1, int((gala['zhi'][igal] - zmin) / zstep))
        iob = min(nz-1, int((gala['z'][igal] - zmin) / zstep))
        S_obs[ilo+1:iob, igal] = 1
        S_vis[ilo+1:ihi, igal] = 1
        Vp = V_int[ilo+1] - afac*cosmo.dm(gala['zlo'][igal])**3
        S_obs[ilo, igal] = Vp/V[ilo]
        S_vis[ilo, igal] = Vp/V[ilo]
        Vp = afac*cosmo.dm(gala['z'][igal])**3 - V_int[iob]
        S_obs[iob, igal] = Vp/V[ihi]
        Vp = afac*cosmo.dm(gala['zhi'][igal])**3 - V_int[ihi]
        S_vis[ihi, igal] = Vp/V[ihi]
    return S_obs, S_vis

def kcorr(z, coeffs):
    """Reconstruct K-correction from SED fit coeffs."""
#     return np.polynomial.polynomial.polyval(z - par['z0'], kcoeff)
    
    if np.ndim(coeffs) == 1:
        k = par['kc'].kcorrect(redshift=z, coeffs=coeffs, band_shift=par['z0'])
        kcorrect = k[par['r_index']]
        
    else :
        kcorrect = np.zeros((len(coeffs), len(z)))
        for i in range(len(coeffs)):
            kcorrect[i] = par['kc'].kcorrect(redshift=z, coeffs=np.broadcast_to(coeffs[i], (len(z), len(coeffs[i]))), band_shift = par['z0'])[:, par['r_index']]

    return kcorrect

def delta_P_solve(Q, gala, zbin, zhist, V, V_int, S, P_prior, nitermax=50, 
                  delta_tol=1e-4, P_tol=1e-3):
    """Solve for LF, density variation and density evolution P for given  
    luminosity evolution parameter Q."""

    converged = False
    niter = 0
    P = 0.0
    mu = 0.0
    nz = len(zbin)
    Npred = np.zeros(nz)
    delta = np.ones(nz)
    if par['idebug'] > 1:
        print('iteration  Q   P    mu    max delta change')
    while (not(converged) and niter < nitermax):
        P_old = P
        mu_old = mu
        delta_old = delta
        Pz = den_evol(zbin, P)
        # pdb.set_trace()
        # Regular and density-corrected Vmax estimates for each galaxy
        V_max = np.dot(Pz * V, S)
        Vdc_max = np.dot(delta * Pz * V, S)

        # Lagrange multiplier mu
        try:
            mu = scipy.optimize.newton(mufunc, mu_old, fprime=muprime, 
                                       args=(V_max, Vdc_max), tol=1e-3)
        except:
            mu = 0.0

        # Predicted mean galaxy number and variance per redshift bin
        for iz in range(nz):
            Npred[iz] = (Pz[iz] * V[iz] * S[iz,:] * gala['weight'] / 
                         (Vdc_max + mu*V_max)).sum()
        den_var = (1 + J3*Npred/V) / Npred

        # Overdensity delta via solution of quadratic eqn (23)
        delta = np.ones(nz)
        arg = (1 - Npred*den_var)**2 + 4*zhist*den_var
        idx = arg >= 0
        delta[idx] = 0.5 * (1 - Npred[idx]*den_var[idx] + arg[idx]**0.5)

        # Solve for density evolution parameter P via eqn (25)
        P = 0.4*ln10*P_prior[1]/nz * np.sum(zbin * (zhist - Npred*(delta+mu)))
        P = min(max(P, -5), 5)

        # Check for convergance
        P_err = abs(P - P_old)
        delta_err = np.max(np.absolute(delta - delta_old))
        if par['idebug'] > 1:
            print(niter, Q, P, mu, delta_err)
        if P_err < P_tol and delta_err < delta_tol:
            converged = True
        niter += 1

    V_max = np.dot(Pz * V, S)
    Vdc_max = np.dot(delta * Pz * V, S)
    V_max_corr = Vdc_max + mu*V_max
    return converged, P, mu, Npred, delta, den_var, Pz, V_max_corr

def delta_solve(P, Q, gala, nz, zminmax, zbin, zhist, V, V_int, S, 
                nitermax=50, delta_tol=1e-4):
    """Solve for overdensity delta for given P, Q."""
    
    (zmin, zmax) = zminmax
    converged = False
    niter = 0
    Npred = np.zeros(nz)
    delta_old = np.ones(nz)
    Pz = den_evol(zbin, P)
    # pdb.set_trace()
    if par['idebug'] > 1:
        print('iteration  max(delta Delta)')
    while (not(converged) and niter < nitermax):
        # Density-corrected Vmax estimates for each galaxy
        Vdc_max = np.dot(delta_old * Pz * V, S)
            
        # Predicted mean galaxy number per redshift bin
        for iz in range(nz):
            Npred[iz] = (Pz[iz] * V[iz] * S[iz,:] * gala['weight'] / Vdc_max).sum()

        # Overdensity = weighted sum of galaxies in bin / predicted
        delta = np.ones(nz)
        occ = Npred > 0
        delta[occ] = zhist[occ]/Npred[occ]

        # Check for convergance
        delta_err = np.max(np.absolute(delta - delta_old))
        if par['idebug'] > 1:
            print(niter, delta_err)
        delta_old = delta
        if delta_err < delta_tol:
            converged = True
        niter += 1
    den_var = (1 + J3*Npred/(V / (u.Mpc**3))) / Npred

    return converged, Npred, delta, den_var, Pz, Vdc_max, niter

def lf1d(gala, V_max_corr, lf_bins):
    """Calculates univariate LF."""
    
    absval = gala['absval_lf']
    Mbin = lf_bins[:-1] + 0.5*np.diff(lf_bins)
    # for i in range(len(Mbin)):
    #     idx = (lf_bins[i] <= absval) * (absval < lf_bins[i+1])
    #     Mbin[i] = np.mean(absval[idx])

    Mhist, edges = np.histogram(absval, lf_bins)
    whist, edges = np.histogram(absval, lf_bins, weights=gala['weight'])
    wt = gala['weight']/V_max_corr
    if par['lf_est'] == 'bin':
        phi, edges = np.histogram(absval, lf_bins, weights=wt)
        phi /= np.diff(lf_bins)
        kde_bandwidth = 0
    if par['lf_est'] == 'kde':
#         kde = pyqt_fit.kde.KDE1D(absval, lower=lf_bins[0], 
#                                  upper=lf_bins[-1], weights=wt)
        kde = gaussian_kde(absval, weights=wt)
        phi = kde(Mbin) * wt.sum()
        kde_bandwidth = kde.bandwidth
    if par['lf_est'] == 'weight':
#         bin_centers = Mbin
#         bin_width = np.diff(lf_bins)[0]

#         weights = np.zeros((len(bin_centers), len(absval)))
#         for i in range(len(bin_centers)):
#             for j in range(len(absval)):
#                 if np.abs(absval[j] - bin_centers[i]) < bin_width:
#                     weights[i, j] = (1 - (np.abs(absval[j] - bin_centers[i]) / bin_width)) * wt[j] * (u.Mpc**3)
#                     if i==0 and absval[j]<min(bin_centers) or i==len(bin_centers) and absval[j]>max(bin_centers):
#                         weights[i, j] = 1 * wt[j] * (u.Mpc**3)
#                 else:
#                     weights[i, j] = 0
#         phi = np.sum(weights, axis=1)
#         kde_bandwidth = 0

        nbins = len(Mbin)
        bin_width = lf_bins[1] - lf_bins[0]
        sel = (lf_bins[0] <= absval) * (absval < lf_bins[-1])
        absval, wt = absval[sel], wt[sel]
        hist, phi = np.zeros(nbins), np.zeros(nbins)
        pf = (absval-lf_bins[0])/bin_width - 0.5
        p = np.floor(pf).astype(int)
        ok = (p >= 0) * (p < nbins-1)
        pstar = pf[ok] - p[ok]
        np.add.at(hist, p[ok], (1-pstar))
        np.add.at(hist, p[ok]+1, pstar)
        np.add.at(phi, p[ok], (1-pstar)*wt[ok] * (u.Mpc**3))
        np.add.at(phi, p[ok]+1, pstar*wt[ok] * (u.Mpc**3))
        first = (p < 0)
        hist[0] += len(wt[first] * (u.Mpc**3))
        phi[0] += np.sum(wt[first] * (u.Mpc**3))
        last = (p >= nbins-1)
        hist[nbins-1] += len(wt[last] * (u.Mpc**3))
        phi[nbins-1] += np.sum(wt[last] * (u.Mpc**3))
        err = phi/hist**0.5
        kde_bandwidth = 0

    # Jackknife errors
    phi_jack = np.zeros((njack, len(phi)))
    for jack in range(njack):
        idx = (gala['ra'] < ra_jack[jack]) + (gala['ra'] >= ra_jack[jack] + 4.0)
        if par['lf_est'] == 'bin':
            phi_jack[jack, :], edges = np.histogram(
                absval[idx], lf_bins, weights=wt[idx])
            phi_jack[jack, :] *= float(njack)/(njack-1)/np.diff(lf_bins)
            phi_err = np.sqrt((njack-1) * np.var(phi_jack, axis=0))
        if par['lf_est'] == 'kde':
#             kde = pyqt_fit.kde.KDE1D(absval[idx], lower=lf_bins[0], 
#                                      upper=lf_bins[-1], weights=wt[idx])
            kde = gaussian_kde(absval[idx], weights=wt[idx])
            phi_jack[jack, :] = kde(Mbin) * wt[idx].sum() * njack / (njack-1)
            phi_err = np.sqrt((njack-1) * np.var(phi_jack, axis=0))
        if par['lf_est'] == 'weight':
            phi_err = err
    lf = {'Mbin': Mbin, 'Mhist': Mhist, 'whist': whist, 
          'phi': phi, 'phi_err': phi_err, 'kde_bandwidth': kde_bandwidth}
    return lf

def den_evol(z, P):
    """Density evolution at redshift z."""
    assert par['ev_model'] in ('z', 'z1z')
    if par['ev_model'] == 'z':
        return 10**(0.4*P*z)
    if par['ev_model'] == 'z1z':
        return 10**(0.4*P*z/(1+z))