#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 16:00:51 2026

@author: esivkova
"""


import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import pandas as pd
import numpy as np
import kepler
import os
import spleaf
import copy
import xml.etree.ElementTree as ET
from kepmodel.astro import AstroModel as AstrometricModel
from scipy.optimize import least_squares
from matplotlib.gridspec import GridSpec

class fitGaia:
    def __init__(self, object_name, DataRelease = 4):
        #self.object_name
        self.object_name = object_name
        self.DataRelease = DataRelease
        
        Trefs = {1:2015.0,
                 2:2015.5,
                 3:2016,
                 4:2017.5,
                 5:2020}
        self.Tref = Time(Trefs[DataRelease],format='decimalyear')
        
    def load_file(self, filename_gaia_astro):
        columns = 'transitid ccd_id obs_time_tcb centroid_pos_al centroid_pos_error_al \
            parallax_factor_al scan_pos_angle outlier_flag'.split()
        self.gaiaastro = pd.read_csv(filename_gaia_astro, delim_whitespace=True, names=columns, comment='#')
        
        # Convenience: precompute scan-angle projection terms
        psi = np.deg2rad(self.gaiaastro["scan_pos_angle"].to_numpy())
        self.gaiaastro["spsi_obs"] = np.sin(psi)
        self.gaiaastro["cpsi_obs"] = np.cos(psi)
        
    def load_dataframe(self, dataframe):
        columns = 'transitid ccd_id obs_time_tcb centroid_pos_al centroid_pos_error_al \
            parallax_factor_al scan_pos_angle outlier_flag'.split()
        df = dataframe.copy()          # <-- IMPORTANT: copy()
        df.columns = columns
        self.gaiaastro = df
        
        # Convenience: precompute scan-angle projection terms
        psi = np.deg2rad(self.gaiaastro["scan_pos_angle"].to_numpy())
        self.gaiaastro["spsi_obs"] = np.sin(psi)
        self.gaiaastro["cpsi_obs"] = np.cos(psi)
        
    def fit_kepmodel(self, data_dir=None):
        
        # from https://github.com/esa/gaia-bhthree
        # Gaia Collaboration 2024, https://doi.org/10.1051/0004-6361/202449763
        
        gaia_astrometry = self.gaiaastro
        
        # filter our unused data
        gaia_astrometry = gaia_astrometry[gaia_astrometry['outlier_flag']!=1]

        # set auxiliary fields
        gaia_astrometry['relative_time_year'] = Time(gaia_astrometry['obs_time_tcb'], format='jd', scale='tcb').jyear - self.Tref.jyear
        gaia_astrometry['mjd'] = Time(gaia_astrometry['obs_time_tcb'], format='jd', scale='tcb').mjd
        gaia_astrometry['relative_time_day'] = gaia_astrometry['relative_time_year'] * u.year.to(u.day)
        gaia_astrometry['cpsi_obs'] = np.cos(np.deg2rad(gaia_astrometry['scan_pos_angle']))
        gaia_astrometry['spsi_obs'] = np.sin(np.deg2rad(gaia_astrometry['scan_pos_angle']))
        
        include_jitter_term = False

        if include_jitter_term:
            astrometric_jitter_value = 0.05
        else:
            astrometric_jitter_value = 0.0
            
        # set up the single-star model with an additional jitter term of 0.05 mas
        single_star_model = AstrometricModel(gaia_astrometry['relative_time_day'].values, 
                                             gaia_astrometry['centroid_pos_al'].values, 
                                             gaia_astrometry['cpsi_obs'].values, 
                                             gaia_astrometry['spsi_obs'].values, 
                                             err=spleaf.term.Error(gaia_astrometry['centroid_pos_error_al'].values),
                                             jit=spleaf.term.Jitter(astrometric_jitter_value))
            
        # define the linear parameters
        single_star_model.add_lin(gaia_astrometry['spsi_obs'].values, 'ra')
        single_star_model.add_lin(gaia_astrometry['cpsi_obs'].values, 'dec')
        single_star_model.add_lin(gaia_astrometry['parallax_factor_al'].values, 'parallax')
        single_star_model.add_lin(gaia_astrometry['relative_time_year'].values * gaia_astrometry['spsi_obs'].values, 'mura')
        single_star_model.add_lin(gaia_astrometry['relative_time_year'].values * gaia_astrometry['cpsi_obs'].values, 'mudec')
        gaia_astrometry['ppfact_obs'] = gaia_astrometry['parallax_factor_al']
        gaia_astrometry['da_mas'] = gaia_astrometry['centroid_pos_al']
        gaia_astrometry['sigma_da_mas'] = gaia_astrometry['centroid_pos_error_al']

        # add jitter term
        if include_jitter_term:
            single_star_model.fit_param += ['cov.jit.sig']

        # perform the fit
        single_star_model.fit()
        
        # params_ss = single_star_model.get_param()
        single_star_model.show_param()
        
        chi2r_ss = np.sqrt(single_star_model.chi2()/(len(gaia_astrometry['da_mas'])-5))
        
        params = single_star_model.get_param_error()
        plx1 = params[0][2]
        plx1_err = params[1][2]
        pmra1 = params[0][3]
        pmra1_err = params[1][3]
        pmdec1 = params[0][4]
        pmdec1_err = params[1][4]
        
        # print(chi2r_ss*plx1_err)
        
        print('chi2r ss', chi2r_ss**2)
        
        model = copy.deepcopy(single_star_model)

        # Periodogram settings 
        Pmin = 3
        Pmax = 10000
        nfreq = 10000
        nu0 = 2 * np.pi / Pmax
        dnu = (2 * np.pi / Pmin - nu0) / (nfreq - 1)

        # compute periodogram
        nu, power = model.periodogram(nu0, dnu, nfreq)

        # convert from angular frequency to period
        P = 2 * np.pi / nu

        # identify highest peak and compute false-alarm probability (FAP)
        kmax = np.argmax(power)
        faplvl = model.fap(power[kmax], nu.max())
        
        keplerian_model = copy.deepcopy(model)
        keplerian_model.add_keplerian_from_period(P[kmax])
        keplerian_model.fit()
        
        # params1 = keplerian_model.get_param()
        param = ['P', 'Tp', 'as', 'e', 'w', 'i', 'bigw']
        keplerian_model.set_keplerian_param(f'0', param=param)
        params = keplerian_model.get_param_error()
        
        keplerian_parameters = {}
        for i, key in enumerate(keplerian_model.keplerian['0']._param):
            keplerian_parameters[key] = keplerian_model.keplerian['0']._par[i]

        linear_parameters = {}
        for i, key in enumerate(keplerian_model._lin_name):
            linear_parameters[key] = keplerian_model._lin_par[i]
            
        chi2r_bs = np.sqrt(keplerian_model.chi2()/(len(gaia_astrometry['da_mas'])-12))
        plx2 = params[0][2]
        plx2_err = params[1][2] *chi2r_bs
        pmra2 = params[0][3]
        pmra2_err = params[1][3] *chi2r_bs
        pmdec2 = params[0][4]
        pmdec2_err = params[1][4] *chi2r_bs
        
        # print(chi2r_bs*plx2_err)
        print('chi2r bs', chi2r_bs**2)
        
        keplerian_parameters = {'a': keplerian_parameters['as'], 
                   'i': np.rad2deg(keplerian_parameters['i']), 
                   'Omega': np.rad2deg(keplerian_parameters['bigw']), 
                   'e':keplerian_parameters['e'],
                   'w':np.rad2deg(keplerian_parameters['w']), 
                   'T0': keplerian_parameters['Tp'], 
                   'P':keplerian_parameters['P'],
                   'plx_ss': plx1,
                   'plxerr_ss': plx1_err,
                   'pmra_ss': pmra1,
                   'pmraerr_ss': pmra1_err,
                   'pmdec_ss': pmdec1,
                   'pmdecerr_ss': pmdec1_err,
                   'plx_bs': plx2,
                   'plxerr_bs': plx2_err,
                   'pmra_bs': pmra2,
                   'pmraerr_bs': pmra2_err,
                   'pmdec_bs': pmdec2,
                   'pmdecerr_bs': pmdec2_err,
                   'chi2r_ss': chi2r_ss,
                   'chi2r_bs': chi2r_bs}
        
        if data_dir is not None:
            np.savetxt(f"{data_dir}/{self.object_name}_DR{str(self.DataRelease)}.txt", np.array(list(keplerian_parameters.items())), fmt="%s")
        
        # print(f"Best-fit parameter (Campbell elements)\nkep.0.P is the period in days, kep.0.as is the semimajor axis in milli-arcseconds\n")
        keplerian_model.show_param()
        
        self.keplerian_model = keplerian_model
        self.linear_parameters = linear_parameters
        
        return keplerian_parameters
    
    def _asymmetric_sine(self, x, Gamma):
        if np.isclose(Gamma, 0.0):
            return np.sin(x)
        return (1.0/Gamma) * np.arctan2(Gamma*np.sin(x), 1.0 - Gamma*np.cos(x))
