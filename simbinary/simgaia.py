#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:54:17 2026

@author: esivkova
"""

from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
import pandas as pd
import numpy as np
import kepler
import os
import spleaf
import copy
import requests
import xml.etree.ElementTree as ET
from kepmodel.astro import AstroModel as AstrometricModel
from scipy.optimize import least_squares
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator

from astropy.timeseries import LombScargle
import json
from datetime import datetime
import io
import re
import contextlib


class SimBinary:
    def __init__(self, ObjectParameters, DataRelease = 4, SaveGost = True, errCCD = False, GaiaPuls = True, perturbation=None):
        """
        Creates SimBinary object and produces epoch astrometry simulation
        Parameters
        ----------
        ObjectParameters : dictionary following supported pattern.
        DataRelease : Gaia DR. Integer from 1 to 5.
        SaveGost : Boolean. Default True. If True creates a folder "gost" in current working directory.
        errCCD : Boolean. Default False. Add per CCD noise.
        GaiaPuls : Boolean. Default True. Use Gaia pulsation for Cepheid or not.
        perturbation : dictionary following supported pattern. Default None.
        """
        
        # filtering nans
        ObjectParameters = {k: ObjectParameters[k] for k in ObjectParameters\
                            if not pd.isnull(ObjectParameters[k])}
        # checking that all parameters are ok
        self._check_params(ObjectParameters, DataRelease)
        # defining few useful parameters
        self.ObjectParameters = ObjectParameters
        self.ObjectName = ObjectParameters['Object']
        self.ObjectType = ObjectParameters['type']
        self.DataRelease = DataRelease
        self.SaveGost = SaveGost
        self.GaiaPuls = GaiaPuls
        self.errCCD = errCCD
        self.has_pulsation = False #default
        self.has_convection = False #default
        self.ra0 = self.ObjectParameters.get('ra0', 0) #default is 0
        self.dec0 = self.ObjectParameters.get('dec0', 0) #default is 0
        
        Trefs = {1:2015.0, # reference time depending on the DR
                 2:2015.5,
                 3:2016,
                 4:2017.5,
                 5:2020}
        self.Tref = Time(Trefs[DataRelease],format='decimalyear')
        
        # check perturbation if any
        self._check_perturbation(perturbation)
        
        # query missing parameters
        self._querySimbadGaia()
        if GaiaPuls and self.ObjectType=='cepheid':
            # query Cepheid pulsation parameters in Gaia DR3
            self._queryGaiaCepheid()
        
        # load gost if already saved, get gost data if not
        gostdata = self.LoadGost()
            
        # apply proper motion (PM) correction if no PM in parameters
        if self.ObjectPMRA is None or self.ObjectPMDEC is None: 
            print('Applying correction for DR3 proper motion...')
            
            # the correction if applied with DR3, so limit observations to DR3 time
            self.LimitGost(gostdata, DR=3)
            
            # set PM to 0
            self.ObjectPMRA, self.ObjectPMDEC = 0, 0
            # simulate along scan of the target with null PM
            w_bs = self.SimWAL(errCCD=False)
            
            # fit single star (SS) model
            mA = np.array([
                np.sin(self.scanAngleRAD),                # alpha0
                self.reltimes*np.sin(self.scanAngleRAD),  # pmra
                np.cos(self.scanAngleRAD),                # delta0
                self.reltimes*np.cos(self.scanAngleRAD),  # pmdec
                self.plxFactorAL                          # parallax
                ]).T
            werr = np.array(len(w_bs)*[self.errALCCD(self.ObjectGmag)])
            Cinv = np.diag(1/werr**2)
            p_fit = np.linalg.solve(mA.T @ Cinv @ mA, mA.T @ Cinv @ w_bs)
            _, pmra, _, pmdec, _ = p_fit
            
            # calculate errors
            F = mA.T @ Cinv @ mA
            Cov_p = np.linalg.inv(F)
            errors = np.sqrt(np.diag(Cov_p))
            _, pmra_err, _, pmdec_err, _ = errors
            
            # get chi2
            w_fit = mA @ p_fit
            chi2r = np.sum(((w_bs-w_fit)/werr)**2)/(len(w_fit)-5)
            
            # print and save the resulting PM
            print(f'Correction: {np.round(pmra*365.25, 3)}\u00B1{np.round(pmra_err*365.25*chi2r**0.5, 3)} '
                  f'{np.round(pmdec*365.25, 3)}\u00B1{np.round(pmdec_err*365.25*chi2r**0.5, 3)}')
            self.ObjectPMRA = self.ObjectPMRA_DR3cat - pmra*365.25    # the fitted PM was in mas/day
            self.ObjectPMDEC = self.ObjectPMDEC_DR3cat - pmdec*365.25 # converting in mas/year
            self.ObjectPMRA_err = self.ObjectPMRA_DR3cat_err + pmra_err*365.25*chi2r**0.5
            self.ObjectPMDEC_err = self.ObjectPMDEC_DR3cat_err + pmdec_err*365.25*chi2r**0.5
            print(f'Proper motion corrected to: '
                  f'{np.round(self.ObjectPMRA, 3)}\u00B1{np.round(self.ObjectPMRA_err,3)} '
                  f'and {np.round(self.ObjectPMDEC, 3)}\u00B1{np.round(self.ObjectPMDEC_err,3)} mas')
        # elif self.ObjectPMDEC is None: # to check what is that
        #     self.ObjectPMRA=self.ObjectPMRA_DR3cat
        #     self.ObjectPMDEC=self.ObjectPMDEC_DR3cat
        
        # limit gost observations to requested DR time
        self.LimitGost(gostdata, DR=self.DataRelease)
        
        if self.errCCD: # add errors in produced along scans
            length = len(self.reltimes)
            self.errors = np.random.normal(0, self.errALCCD(self.ObjectGmag), length)
        
        if self.ObjectType == 'AGB': # a star with convection case
            self.LightCurve()
            self.Convection()
            self.has_convection = True
        
        # final simulation of along scans
        w_bs = self.SimWAL(errCCD=errCCD)
                
        
    def _check_params(self, ObjectParameters, DataRelease):
        """
        Internal function to check if the parameters correspond to supported format.
        Parameters
        ----------
        ObjectParameters : a dictionary following requested scheme
        DataRelease : 1, 2, 3, 4 or 5, will repot error instead
        """
        if DataRelease not in [1, 2, 3, 4, 5]:
            raise ValueError("Please, choose on of the Gaia DR: 1, 2, 3, 4, 5. \
                             The current value '{DataRelease} is not supported.'")
        
        schema = { # parameter's name, authorized type, etc
             'Object': {'required': True, 'type': str},
             'type':   {'required': True, 'type': str, 'selection':['BH', 'binary', 'cepheid', 'exoplanet', 'AGB']},
             'ra':     {'required': False,'type': (float, int, np.floating), 'range': [0, 360]},
             'dec':    {'required': False,'type': (float, int, np.floating), 'range': [-90, 90]},
             'id3':    {'required': False,'type': str},
             'P':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 10e6]},
             'a':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 160]},
             'e':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 1]},
             'i':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 180]},
             'Omega':  {'required': True, 'type': (float, int, np.floating), 'range': [0, 360]},
             'w':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 360]},
             'T0':     {'required': True, 'type': (float, int, np.floating)},
             'q':      {'required': True, 'type': (float, int, np.floating), 'range': [0, 10e3]},
             'plx':    {'required': True, 'type': (float, int, np.floating), 'range': [0, 10e3]},
             'pmra':   {'required': False,'type': (float, int, np.floating)},
             'pmdec':  {'required': False,'type': (float, int, np.floating)},
             'Ppuls':  {'required': False,'type': (float, int, np.floating), 'range': [0, 10e6]},
             'T0puls': {'required': False,'type': (float, int, np.floating)},
             'Vmax':   {'required': False,'type': (float, int, np.floating)},
             'Vmin':   {'required': False,'type': (float, int, np.floating)},
             'Vtot':   {'required': False,'type': (float, int, np.floating)},
             'Vcomp':  {'required': False,'type': (float, int, np.floating)},
             'fratio': {'required': False,'type': (float, int, np.floating)},
             'ra0':    {'required': False,'type': (float, int, np.floating)},
             'dec0':   {'required': False,'type': (float, int, np.floating)}
        }
        
        for key, rule in schema.items():
            # check if required
            if rule['required'] and key not in ObjectParameters:
                raise KeyError(f"The parameter '{key}' is missing. Please, add \
                               it to the parameters dictionary.")

            if key in ObjectParameters:
                expected = rule["type"]
                value = ObjectParameters[key]
                # check the type
                if not isinstance(value, expected):
                    raise TypeError(f"The parameter '{key}' is {type(ObjectParameters[key])}.\
                                    Should be {expected}.")
                # check if corresponds to authorized range
                if 'range' in rule:
                    min_v, max_v = rule['range']
                    if not (min_v <= value <= max_v):
                        raise ValueError(f"The parameter '{key}' is out of range \
                                         ({min_v}, {max_v}) with value {value}.")
                # check if corresponds to authorized range
                if 'selection' in rule:
                    if value not in rule['selection']:
                        raise ValueError(f"The object type '{value}' doesn\'t \
                                         correspond to the supported ones: {rule['selection']}")
                                         
    def _check_perturbation(self, perturbation):
        """
        Internal function to check the perturbation format
        Parameters
        ----------
        perturbation : the perturbation dictionary
        """
        if perturbation:
        
            if 'component' not in perturbation:
                raise KeyError("The parameter 'component' is missing. Please, add it to the perturbation dictionary alongside the perturbation array/function labeled as 'value'.")
            if 'value' not in perturbation:
                raise KeyError("The parameter 'value' is missing. Please, add it to the perturbation dictionary")
                
            if perturbation['component'] not in [1,2]:
                raise ValueError(f"Please, choose component 1 or 2. Current component is {perturbation['component']}.")
                
            if not callable(perturbation['value']) and not isinstance(perturbation['value'], (list, tuple, np.ndarray)):
                raise ValueError(f"Please, provide a perturbation of function or array type. Current type is {type(perturbation['value'])}.")
            
            if isinstance(perturbation['value'], (list, tuple, np.ndarray)):
                arr = np.asarray(perturbation['value'])
                # check if 2D array
                if arr.ndim != 2:
                    raise ValueError("Perturbation must be a 2D array with shape (2, N)")
                if arr.shape[0] == 2:
                    pass
                elif arr.shape[1] == 2:
                    arr = arr.T # transpose if (N, 2)
                else:
                    raise ValueError(
                        "Perturbation must have shape (2, N) or (N, 2), "
                        f"(got {arr.shape})"
                    )
                perturbation['value'] = arr
            
            print(f"Perturbation for component {perturbation['component']}")
        self.perturbation = perturbation
    
    def errALCCD(self, G):
        """
        Adopted from gaiamock https://github.com/kareemelbadry/gaiamock/
        El-Badry et al. 2024, 2024OJAp....7E.100E
        Parameters
        ----------
        G: object's magnitude in band G

        Returns
        -------
        per CCD error in mas (float)
        """
        G_vals =    [ 4,    5,   6,     7,   8.2,  8.4, 10,    11,    12,  13,    14,   15,   16,   17,   18,   19,  20]
        sigma_wal = [0.4, 0.35, 0.15, 0.17, 0.23, 0.13,0.13, 0.135, 0.125, 0.13, 0.15, 0.23, 0.36, 0.63, 1.05, 2.05, 4.1]
        return np.interp(G, G_vals, sigma_wal)
    
    def _querySimbadGaia(self):
        """
        Internal function to query the Gaia and Simbad database if missing requested parameters
        """
        self.ObjectRA = None
        self.ObjectDEC = None
        self.id3 = None 
        self.ObjectPMRA = None
        self.ObjectPMDEC = None
        self.ObjectGmag = None
        
        if 'id3' in self.ObjectParameters:
            self.id3 = self.ObjectParameters['id3']
        
        if 'ra' in self.ObjectParameters and 'dec' in self.ObjectParameters:
            self.ObjectRA = self.ObjectParameters['ra']
            self.ObjectDEC = self.ObjectParameters['dec']
            
        if 'pmra' in self.ObjectParameters and 'pmdec' in self.ObjectParameters:
            self.ObjectPMRA = self.ObjectParameters['pmra']
            self.ObjectPMDEC = self.ObjectParameters['pmdec']
        
        if 'Vtot' in self.ObjectParameters:
            self.ObjectGmag = self.ObjectParameters['Vtot']
        
        if self.id3 is None:
            Simbad.add_votable_fields('ids')
            result = Simbad.query_object(self.ObjectName)
            
            if len(result) == 0:
                raise ValueError('The object was not resolved by Simbad. \
                    Try with to change the target name or to add RA, \
                        DEC, VRA, VDEC and GAIA DR3 ID to avoid Simbad query.')
            
            if self.id3 is None:
                # query Simbad to get Gaia DR3 id
                ids = result['ids'][0].split('|')
                gaia_id = [s for s in ids if 'Gaia DR3' in s]
                if len(gaia_id) == 0:
                    raise ValueError('The object is not in DR3.')
                # Take only the number
                self.id3 = gaia_id[0][9:]
                print('Gaia DR3 ID added with Simbad')
                
        if None in [self.ObjectRA, self.ObjectPMRA, self.ObjectGmag]:
            # query Gaia DR3 based on id
            Gaia.ROW_LIMIT = 1  
            query = f"""
            SELECT *
            FROM gaiadr3.gaia_source
            WHERE source_id = {self.id3}
            """
            job = Gaia.launch_job(query)
            object_data = job.get_results()
            
            if self.ObjectRA is None:
                self.ObjectRA = object_data['ra'].data[0]
                self.ObjectDEC = object_data['dec'].data[0]
                print('RA/DEC coordinates added with Gaia DR3')
            if self.ObjectPMDEC is None:
                self.ObjectPMRA_DR3cat = object_data['pmra'].data[0]
                self.ObjectPMDEC_DR3cat = object_data['pmdec'].data[0]
                self.ObjectPMRA_DR3cat_err = object_data['pmra_error'].data[0]
                self.ObjectPMDEC_DR3cat_err = object_data['pmdec_error'].data[0]
                print('Proper motion RA/DEC added with Gaia DR3')
            if self.ObjectGmag is None:
                self.ObjectGmag = object_data['phot_g_mean_mag'].data[0]
                print('Gmag added with Gaia DR3')
            
    def LoadGost(self):
        """
        Load Gost data for object's coordinates
        Returns
        -------
        Dataframe with Gost data (obs dates, scan angles, parallax factors)
        """
        name = self.ObjectName.replace(' ', '_') # avoid space
        filepath = f"gost/gost_{name}.csv"
        # check if the file already exists
        if os.path.isfile(filepath):
            gostdata = pd.read_csv(filepath, sep=',') # load the existing file
        else:
            # if not, download Gost data
            gostdata = self.DownloadGost(self.ObjectRA, self.ObjectDEC, self.ObjectName)
            if self.SaveGost:
                # if save Gost, save the data in "gost" folder of current working directory
                if not os.path.isdir('gost/'):
                    os.mkdir('gost/')
                print(f"Downloading GOST data to {os.getcwd()+'/gost'}.")
                gostdata.to_csv(filepath, sep =',')
        return gostdata
    
    def LimitGost(self, gostdata, DR):
        """
        Limit in time Gost data depending on the Data Release
        Parameters
        ----------
        gostdata : Dataframe with Gost data
        DR : Data Release (from 1 to 5)
        """
        TstopDRs = {1:'2015-09-16T00:00:00', # Gaia data releases data acquisition limits
                 2:'2016-05-23T11:35:00',    # from Gaia documentation
                 3:'2017-05-28T08:44:00',
                 4:'2020-01-20T22:00:00',
                 5:'2025-01-16T00:00:00'}
        Tstop = Time(TstopDRs[DR],format='isot')
        
        gostdata = gostdata[gostdata['ObservationalTimeBarycentre']<Tstop.jd]
        
        self.times = Time([Time(t, format='jd') \
                           for t in gostdata['ObservationalTimeBarycentre']])
        
        self.scanAngleRAD = gostdata['scanAngle']
        self.plxFactorAL = gostdata['parallaxFactorAL']
        self.plxFactorAC = gostdata['parallaxFactorAC']
        
        self.scanAngleDEG = np.rad2deg(self.scanAngleRAD)
        self.reltimes = (self.times - self.Tref).to(u.day).value
        self.timesjd = self.times.to_value('jd')
        
                    
    def DownloadGost(self, ra, dec, target_name):
        """
        Download Gost data from https://gaia.esac.esa.int/gost/ based on object's coordinates.
        Adapted from Download_HIP_Gaia_GOST by Yicheng Rui:
        https://github.com/ruiyicheng/Download_HIP_Gaia_GOST/tree/main
        Parameters
        ----------
        ra : float
        dec : float
        target_name : string

        Returns
        -------
        Dataframe with Gost data
        """
        url = f"https://gaia.esac.esa.int/gost/GostServlet?ra={str(ra)}+&dec={str(dec)}"

        with requests.Session() as s:
            s.get(url)
            headers = {"Cookie": f"JSESSIONID={s.cookies.get_dict()['JSESSIONID']}"}
            response = s.get(url, headers=headers, timeout=1000)#,proxies=proxies)
        root = ET.fromstring(response.text)
        columns = ["Target", "CcdRow", "scanAngle", "parallaxFactorAL", 
                   "parallaxFactorAC", "ObservationalTimeBarycentre"]
        rows = []
        name = root.find('./targets/target/name').text

        for event in root.findall('./targets/target/events/event'):
            details = event.find('details')
            ccdRow = details.find('ccdRow').text
            scanAngle = details.find('scanAngle').text
            parallaxFactorAl = details.find('parallaxFactorAl').text
            parallaxFactorAc = details.find('parallaxFactorAc').text
            observationTimeAtBarycentre = event.find('eventTcbBarycentricJulianDateAtBarycentre').text
            rows.append([name, ccdRow, scanAngle, parallaxFactorAl, parallaxFactorAc, observationTimeAtBarycentre])
        data = pd.DataFrame(rows, columns=columns)
        data = data.astype({"Target": str, "CcdRow": int,"scanAngle": float,"parallaxFactorAL": float,"parallaxFactorAC": float,"ObservationalTimeBarycentre": float })
        data['Target']=[target_name]*len(data)

        return data
    
    def LightCurve(self):
        """
        Create synthetic ASAS-like light curve (irregular sampling + noise).
        Stored in self.lc as a pandas DataFrame.
        Developed by Leen Decin.
        """
        Vmin = self.ObjectParameters['Vmin']
        Vmax = self.ObjectParameters['Vmax']
        Ppuls = self.ObjectParameters['Ppuls']
        T0 = self.ObjectParameters['T0puls']
        Gamma = self.ObjectParameters.get('Gamma', 0.0)
        seed = self.ObjectParameters.get('seed_conv', 1234)
        rng = np.random.default_rng(seed)

        # You can expose these as ObjectParameters if you like
        Nlc = self.ObjectParameters.get('Nlc', 350)          # typical few hundred points
        season_length = self.ObjectParameters.get('LC_season_days', 220)  # observing season per year
        cadence_min = self.ObjectParameters.get('LC_cadence_min', 1.0)    # days
        cadence_max = self.ObjectParameters.get('LC_cadence_max', 6.0)    # days
        sigma_floor = self.ObjectParameters.get('LC_sigma_floor', 0.02)   # mag
        sigma_ceiling = self.ObjectParameters.get('LC_sigma_ceiling', 0.08)# mag

        t_start = float(np.min(self.timesjd))
        t_end   = float(np.max(self.timesjd))
        baseline = t_end - t_start
        
        # Build seasons: every ~365d, observe for 'season_length' days with random start offset
        years = int(np.ceil(baseline / 365.25)) + 1
        season_starts = t_start + np.arange(years)*365.25 + rng.uniform(-30, 30, size=years)

        lc_times = []
        for s0 in season_starts:
            s1 = s0 + season_length
            # draw a random-cadence "walk" through the season
            t = s0 + rng.uniform(0, 2)  # small random offset
            while t < s1 and t < t_end:
                if t >= t_start:
                    lc_times.append(t)
                t += rng.uniform(cadence_min, cadence_max)

        lc_times = np.array(lc_times, dtype=float)

        # If we got too many/few points, thin or top up with uniform draws within seasons
        if len(lc_times) > Nlc:
            lc_times = rng.choice(lc_times, size=Nlc, replace=False)
        elif len(lc_times) < max(30, int(0.6*Nlc)):
            # top up: sample uniformly within the overall baseline (still "random-ish")
            extra = rng.uniform(t_start, t_end, size=(Nlc - len(lc_times)))
            lc_times = np.concatenate([lc_times, extra])

        lc_times = np.sort(lc_times)

        # Evaluate asymmetric-sine flux model at lc_times (same shape as your f_AGB definition)
        x_lc = 2*np.pi*(lc_times - T0)/Ppuls
      
        if np.isclose(Gamma, 0.0):
            factor_lc = np.sin(x_lc)
        else:
            factor_lc = (1.0/Gamma) * np.arctan((Gamma*np.sin(x_lc)) / (1.0 - Gamma*np.cos(x_lc)))

        # --- magnitude-domain pulsation model ---
        # Define mag amplitude from Vmin/Vmax (peak-to-peak = Vmin - Vmax)
        A_mag = 0.5 * (Vmin - Vmax)          # semi-amplitude in mag
        m0    = 0.5 * (Vmin + Vmax)          # mean magnitude (midpoint)

        # Gamma=0 => pure sinusoid in magnitude
        mag_true = m0 + A_mag * factor_lc

        # note: sinusoidal in magnitude is not sinusoidal in flux!

        # Per-point uncertainties + noise (heteroscedastic)
        mag_err = rng.uniform(sigma_floor, sigma_ceiling, size=len(lc_times))
        mag_obs = mag_true + rng.normal(0.0, mag_err)

        # Save
        self.lc = pd.DataFrame({
            "time_jd": lc_times,
            "mag_true": mag_true,
            "mag": mag_obs,
            "mag_err": mag_err,
            "band": "V"
        })
        print('self has now light curve data')

        self.has_lightcurve = True
 
    def FluxRatio(self, times, Tplot=0):
        """

        Parameters
        ----------
        times : array-like.
        Tplot : float. Default is 0.

        Returns
        -------
        DataFrame containing flux ratio between components.
        """
        
        fdata = pd.DataFrame(columns=['r1', 'r2'])
        # case if using GaiaPuls
        if self.ObjectType == 'cepheid' and self.GaiaPuls:
            puls, puls_ceph, r1, r2, r1_nps, r2_nps, f_tot = self.addPulsationGaia(times, Tplot)
            fdata['puls'] = puls
            fdata['puls_ceph'] = puls_ceph
            fdata['r1'] = r1
            fdata['r2'] = r2
            fdata['r1_nps'] = r1_nps
            fdata['r2_nps'] = r2_nps
            fdata['f_tot'] = f_tot
            self.has_pulsation = True
        # case if using sinus-like or asymmetric sinus-like pulsation
        elif self.ObjectType == 'cepheid' or self.ObjectType == 'AGB':
            required = ['Vmin', 'Vmax', 'Vcomp', 'Ppuls', 'T0puls']
            if all(key in self.ObjectParameters for key in required):
                puls, puls_ceph, r1, r2, r1_nps, r2_nps, f_tot = self.addPulsation(times, Tplot)
                fdata['puls'] = puls
                fdata['puls_ceph'] = puls_ceph
                fdata['r1'] = r1
                fdata['r2'] = r2
                fdata['r1_nps'] = r1_nps
                fdata['r2_nps'] = r2_nps
                fdata['f_tot'] = f_tot
            else:
                raise KeyError("Please, provide the next parameters to \
                               approximate Cepheid pulsation: Vmin, Vmax, Vcomp, Ppuls, T0puls")
            self.has_pulsation = True
            
        elif self.ObjectType == 'binary':
            # calculate the magnitude of the star number 1 (main) knowing the total (ObjectGmag from Gaia) and companion's magnitudes
            mag_main = - 2.5*np.log10(10**(-0.4*(self.ObjectGmag))-10**(-0.4*(self.ObjectParameters['Vcomp'])))
            f_main = 10**(-0.4*(mag_main-self.ObjectGmag)) # get main's flux taking total as reference
            f_comp = 10**(-0.4*(self.ObjectParameters['Vcomp']-self.ObjectGmag)) # get companion's flux taking total as reference
            
            fdata['r1'] = [f_main/(f_comp+f_main)] # main star
            fdata['r2'] = [f_comp/(f_comp+f_main)] # companion
            
        elif self.ObjectType == 'BH':
            fdata['r1'] = [1] # Star
            fdata['r2'] = [0] # BH
            
        elif self.ObjectType == 'exoplanet':
            fdata['r1'] = [1] # host star
            fdata['r2'] = [0] # exoplanet
            
        else:
            raise KeyError(f"Please, select between: binary, cepheid, BH, AGB or exoplanet. Current is {self.ObjectType}.")
        
        return fdata
        
    def addPulsation(self, times, Tplot):
        """
        Gives flux ratios considering the pulsation (sin or asymm sin cases)
        Parameters
        ----------
        times : array-like.
        Tplot : float. Use 0 in default case.

        Returns
        -------
        puls, puls_ceph, r1, r2, r1_nps, r2_nps, f_tot
        7 arrays of length len(times)
        """
            
        Vmin = self.ObjectParameters['Vmin']
        Vmax = self.ObjectParameters['Vmax']
        Vcomp = self.ObjectParameters['Vcomp']
        Ppuls = self.ObjectParameters['Ppuls']
        T0 = self.ObjectParameters['T0puls']
        T0 = T0 - self.Tref.jd # converting T0 with in a correct time reference (Gaia's)
        T0 = T0 - Tplot
        
        Gamma = self.ObjectParameters.get('Gamma', 0.0) # take 0 if Gamma is not defined
        
        # minimal cepheid flux and companion flux relative to ref flux
        # F/Fref = 10**(-0.4*(m-mref))
        # reference magnitude
        Vref = self.ObjectGmag #np.mean([Vmax, Vmin])
        f_comp = 10**(-0.4*(Vcomp-Vref))
        
        x = 2*np.pi*(times - T0)/Ppuls  # phase argument; use -T0 (time of reference)
        if np.isclose(Gamma, 0.0):
            factor = np.sin(x)
        else:
            factor = (1.0/Gamma) * np.arctan2(Gamma*np.sin(x), 1.0 - Gamma*np.cos(x))
        
        puls = (Vmax-Vmin)/2 * factor + (Vmax+Vmin)/2
        
        puls_ceph = - 2.5*np.log10(10**(-0.4*(puls))-10**(-0.4*(Vcomp)))
        f_ceph = 10**(-0.4*(puls_ceph-Vref))
        # flux ratio for each component
        r1 = f_ceph/(f_comp+f_ceph) # cepheid
        r2 = f_comp/(f_comp+f_ceph) # companion
        
        f_tot = f_ceph+f_comp
        
        # binary system without the pulsating component, non-pulsating system (nps)
        # f_mean = np.mean([f_max, f_min])
        f_mean = 10**(-0.4*(Vref-Vmax))*np.ones(len(puls))
        # f_mean = np.mean(f_ceph)
        r1_nps = f_mean/(f_comp+f_mean) # cepheid
        r2_nps = f_comp/(f_comp+f_mean) # companion
        
        return puls, puls_ceph, r1, r2, r1_nps, r2_nps, f_tot
    
    def _queryGaiaCepheid(self):
        """
        Gives pulsation parameters provided with Gaia DR3 Cepheid catalogue.
        Returns
        -------
        DataFrame with pulsation parameters
        """
        # query Gaia DR3 Cepheid catalog
        Gaia.ROW_LIMIT = 1  
        query = f"""
        SELECT *
        FROM gaiadr3.vari_cepheid
        WHERE source_id = {str(self.id3)}
        """
        job = Gaia.launch_job(query)
        cepheid_data = job.get_results()
        
        if len(cepheid_data) == 0:
            raise ValueError('The object was not found in gaiadr3.vari_cepheid.\
                             You can add the pulsation parameters manually.')
        # get pulsating parameters from the catalog
        A0 = cepheid_data['int_average_g'].data[0]
        N = cepheid_data['num_harmonics_for_p1_g'].data[0] 
        As = cepheid_data['fund_freq1_harmonic_ampl_g'][0][:N].compressed()
        phis = cepheid_data['fund_freq1_harmonic_phase_g'][0][:N].compressed()
        # f1 = cepheid_data['fund_freq1'].data[0]
        T0 = cepheid_data['reference_time_g'].data[0]
        T0 = T0 + Time(2010,format='decimalyear').jd # converting T0 to jd, 2016 DR3 ref
        T0 = T0 - self.Tref.jd # converting T0 with a correct time reference
        P = cepheid_data['pf'].data[0]
        if np.ma.is_masked(P): # if no "pf" use "p1_o" (sometimes it's missing)
            P = cepheid_data['p1_o'].data[0]
            print('The p1_o period used')
        if np.ma.is_masked(P): # if no "pf" and "p1_o" use 1/"fund_freq1" (just in case)
            P = 1/cepheid_data['fund_freq1'].data[0]
            print('The 1/fund_freq period used')

        # save pulsation parameters to a dataframe
        self.PulsParams = pd.DataFrame(columns=['A0', 'N', 'As', 'phis', 'T0', 'P'])
        
        self.PulsParams['As'] = np.array(As)
        self.PulsParams['phis'] = np.array(phis)
        self.PulsParams['A0'] = np.array(A0)
        self.PulsParams['N'] = np.array(N)
        self.PulsParams['T0'] = np.array(T0)
        self.PulsParams['P'] = np.array(P)
        
    def addPulsationGaia(self, times, Tplot=0):
        """
        Gives flux ratios considering the pulsation provided with Gaia DR3.
        Parameters
        ----------
        times : array-like.
        Tplot : float. Use 0 in default case.
        Returns
        -------
        puls, puls_ceph, r1, r2, r1_nps, r2_nps, f_tot
        7 arrays of length len(times)
        """

        # get pulsation parameters
        T0 = self.PulsParams['T0'][0] - Tplot
        A0 = self.PulsParams['A0'][0]
        N = self.PulsParams['N'][0]
        As = self.PulsParams['As'].values
        phis = self.PulsParams['phis'].values
        P = self.PulsParams['P'][0]

        # get an array describing the pulsation
        k = np.arange(1,N+1)
        arg = 2*np.pi*k[:, None]*(times - T0)/P + phis[:, None]
        puls = A0 + np.sum(As[:, None]*np.cos(arg), axis=0)
        
        self.ObjectParameters['Ppuls'] = P # add the pulsation period to the parameters dictionary

        # get only the cepheid contribution
        puls_ceph = - 2.5*np.log10(10**(-0.4*(puls))-10**(-0.4*(self.ObjectParameters['Vcomp'])))
        f_ceph = 10**(-0.4*(puls_ceph-self.ObjectGmag))
        f_comp = 10**(-0.4*(self.ObjectParameters['Vcomp']-self.ObjectGmag))
        f_mean = np.mean(f_ceph)*np.ones(len(puls))
        
        f_tot = f_ceph+f_comp # total flux
        
        # flux fraction for each component  
        r1 = f_ceph/(f_comp+f_ceph) # cepheid
        r2 = f_comp/(f_comp+f_ceph) # companion
        
        # binary system without the pulsating component, non-pulsating system (nps)
        r1_nps = f_mean/(f_comp+f_mean) # cepheid
        r2_nps = f_comp/(f_comp+f_mean) # companion

        return puls, puls_ceph, r1, r2, r1_nps, r2_nps, f_tot
    
    def addConvectionJitter(self, times):
        """
        Provides RA and DEC positions of a convective star's photocentre
        Developed by Leen Decin.
        Parameters
        ----------
        times : array-like.

        Returns
        -------
        Two array-like.
        RA and DEC of the AGB star's photocentre
        """
        
        if times is self.reltimes:
            # if the times is the default ones, no need for interpolation
            return self.dra_conv_mas, self.ddec_conv_mas

        # interpolation is used for plotting

        # binning to have the same value per observation
        sim_astrometry = pd.DataFrame()
        sim_astrometry['times'] = self.reltimes
        sim_astrometry['ra'] = self.dra_conv_mas
        sim_astrometry['dec'] = self.ddec_conv_mas

        dt_sim = np.diff(sim_astrometry['times'])
        new_bin = dt_sim >= 4

        group = np.zeros_like(sim_astrometry['times'], dtype=int)
        group[1:] = np.cumsum(new_bin)

        sim_astrometry['group1'] = group
        bin_sim = sim_astrometry.groupby("group1").agg('mean')
        
        # interpolating
        
        conv = np.array([bin_sim['ra'], bin_sim['dec']])

        # cs = CubicSpline(bin_sim['times'], conv, axis=1)
        # interp = cs(times)
        # fit_dra_conv_mas, fit_ddec_conv_mas = interp

        pchip = PchipInterpolator(bin_sim['times'], conv, axis=1)
        interp = pchip(times)
        fit_dra_conv_mas, fit_ddec_conv_mas = interp
        
        return fit_dra_conv_mas, fit_ddec_conv_mas
        
    def Convection(self):
        
        """
        Create convection/inhomogeneity photocentre jitter for the *primary*.
        Developed by Leen Decin.
        Parameters
        ----------
        sigma_AU: stationary 1D std of photocentre offset in AU (typical 0.08–0.2 AU ballpark).
        tau_days: correlation time in days (months-to-years; try 100–400 d first).
        phase_mod_amp: optional coupling to pulsation phase (0 = none).
                    If >0 and pulsation params exist, sigma is modulated by (1 + A*cos(phase)).

        """
        
        R_star = self.ObjectParameters.get('R_star', 1)
        Ppuls = self.ObjectParameters.get('Ppuls', 200)
        tau_days = self.ObjectParameters.get('tau_conv_days', 200.0)
        seed     = self.ObjectParameters.get('seed_conv', 1234)
        phaseA   = self.ObjectParameters.get('conv_phase_amp', 0.0)
        
        sigma_Rstar = 10**(np.log10(np.log10(Ppuls)/3.38) / 0.12) # from Eq. 8 of Beguin, Chiavassa et al. 2024
        sigma_AU = sigma_Rstar * R_star
        print('sigma_Rstar is', sigma_Rstar, 'Rstar')
        print('sigma_AU is', sigma_AU, 'au')
        
        sigma_mas = sigma_AU * self.ObjectParameters['plx']
        print('sigma_mas is', sigma_mas, 'mas')
        
        rng = np.random.default_rng(seed)

        # times in days relative to Tref (your reltimes is in days)
        t_days = np.asarray(self.reltimes, dtype=float)

        # Optional: let jitter be larger near maximum expansion (simple toy coupling)
        sig = float(sigma_AU)
        if phaseA != 0.0 and ('Ppuls' in self.ObjectParameters) and ('T0puls' in self.ObjectParameters):
            Ppuls = self.ObjectParameters['Ppuls']
            T0puls = self.ObjectParameters['T0puls']
            T0 = self.ObjectParameters['T0']
            T0 = T0 - self.Tref.jd # converting T0 with a correct time reference
            # your timesjd exists; use it for phase if available
            phase = 2*np.pi*(self.reltimes - T0puls)/Ppuls
            mod = 1.0 + float(phaseA)*np.cos(phase)
            mod = np.clip(mod, 0.1, None)  # avoid negative sigma
        else:
            mod = 1.0

        # Generate OU in AU
        dra_AU  = self._ou_process(t_days, sig, tau_days, rng) * mod
        ddec_AU = self._ou_process(t_days, sig, tau_days, rng) * mod

        # Convert AU -> mas using parallax:
        # 1 AU subtends parallax angle; if pll is in mas, delta(mas) = delta(AU) * pll(mas)
        pll_mas = float(self.ObjectParameters['plx'])
        dra_conv_mas  = dra_AU  * pll_mas
        ddec_conv_mas = ddec_AU * pll_mas
        # the mean is not always around zero - shift for that
        self.dra_conv_mas = dra_conv_mas - np.mean(dra_conv_mas)
        self.ddec_conv_mas = ddec_conv_mas - np.mean(ddec_conv_mas)
    
    def _ou_process(self, t_days, sigma, tau_days, rng, burnin_steps=0):
        """Generate an Ornstein–Uhlenbeck (OU) process at (possibly) irregular times.
        # An Ornstein–Uhlenbeck (OU) process is a continuous-time stochastic process that describes 
        # random motion with a tendency to relax back to a preferred value. It is often summarized as 
        # “mean-reverting Brownian motion.”
        Developed by Leen Decin.

        Parameters
        ----------
        t_days : array-like
            Time samples in days (must be non-decreasing).
        sigma : float
            *Stationary* 1D standard deviation of the OU process (same units as output).
        tau_days : float
            Correlation timescale in days. If <=0, falls back to white noise.
        rng : numpy.random.Generator
            Random generator.
        burnin_steps : int, optional
            If >0, evolve the OU process for this many extra steps of size ~median(dt)
            before returning (rarely needed if we start stationary).

        Notes
        -----
        Exact OU discretisation at each step:
            x_k = a x_{k-1} + eps,  a = exp(-dt/tau)
            eps ~ N(0, sigma^2 (1-a^2))
        """
        t_days = np.asarray(t_days, dtype=float)
        x = np.zeros_like(t_days)

        n = len(t_days)
        if n == 0:
            return x

        if tau_days <= 0:
            x[:] = rng.normal(0.0, sigma, size=n)
            return x

        # Start in the stationary distribution (this is the key fix)
        x_prev = rng.normal(0.0, sigma)

        # Optional burn-in (usually unnecessary now, but harmless)
        if burnin_steps and n > 1:
            dts = np.diff(t_days)
            dt0 = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 1.0
            a0 = np.exp(-dt0 / tau_days)
            s0 = sigma * np.sqrt(1.0 - a0*a0)
            for _ in range(int(burnin_steps)):
                x_prev = a0 * x_prev + rng.normal(0.0, s0)

        x[0] = x_prev
        for k in range(1, n):
            dt = t_days[k] - t_days[k-1]
            if dt < 0:
                raise ValueError("t_days must be non-decreasing")

            a = np.exp(-dt / tau_days)
            s = sigma * np.sqrt(1.0 - a*a)
            x_prev = a * x_prev + rng.normal(0.0, s)
            x[k] = x_prev

        return x

    def orbit(self, theta, times): # orbit model
        """
        Orbit function with Campbell.
        Parameters
        ----------
        theta : dictionary containing P, a, e, i (deg), Omega (deg), w (deg) and T0
        times : array-like

        Returns
        -------
        RA and DEC arrays of the orbit
        """
        
        to_rad = (u.deg).to(u.rad)
    
        P = theta['P']
        a = theta['a']
        e = theta['e']
        i = theta['i']*to_rad
        Omega = theta['Omega']*to_rad
        omega = theta['w']*to_rad
        T = theta['T0']

        M = 2*np.pi/P*(times-T)
        E, cosE, sinE = kepler.kepler(M, e)
        
        nu = 2*np.arctan2((1+e)**0.5*np.sin(E/2),(1-e)**0.5*np.cos(E/2))
        r = a*(1-e**2)/(1+e*np.cos(nu)) 
        
        delt_ra = r*(np.sin(Omega)*np.cos(omega+nu) + np.cos(i)*np.cos(Omega)*np.sin(omega+nu)) 
        delt_dec = r*(np.cos(Omega)*np.cos(omega+nu) - np.cos(i)*np.sin(Omega)*np.sin(omega+nu))
        return np.array([delt_ra, delt_dec])

    def SimGaia(self, times, fdata, factra, factdec):
        """
        Simulate Gaia epoch astrometry
        Parameters
        ----------
        times : array-like
        fdata : dataframe. Flux ratio information from flux_ratio
        factra : array-like. Parallax factors of RA
        factdec : array-like. Parallax factors of DEC

        Returns
        -------
        DataFrame
        """
        # primary star/BH parameters, w+pi because it is on the opposite side to companion
        q_comp = self.ObjectParameters['q']/(1+self.ObjectParameters['q'])
        ang1 = 180
        ang2 = 0
        if self.ObjectType == 'BH' or self.ObjectType == 'exoplanet': 
            # companion parameters, q=1 in case of BH because the orbit is already photocentric
            q_comp = 1
            ang1 = 0
            ang2 = 180
            
        params1 = {'a': self.ObjectParameters['a']*q_comp, 
                   'i': self.ObjectParameters['i'], 
                   'Omega': self.ObjectParameters['Omega'], 
                   'e':self.ObjectParameters['e'],
                   'w':(self.ObjectParameters['w']+ang1), 
                   'T0': ((self.ObjectParameters['T0']-self.Tref.jd)*u.day).value, 
                   'P':self.ObjectParameters['P']}
        
        params2 = {'a': self.ObjectParameters['a']*1/(1+self.ObjectParameters['q']), 
                   'i': self.ObjectParameters['i'], 
                   'Omega': self.ObjectParameters['Omega'], 
                   'e':self.ObjectParameters['e'],
                   'w':(self.ObjectParameters['w']+ang2), 
                   'T0': ((self.ObjectParameters['T0']-self.Tref.jd)*u.day).value, 
                   'P':self.ObjectParameters['P']}
        
        # convert pm to mas/day
        pmra = self.ObjectPMRA/365.25
        pmdec = self.ObjectPMDEC/365.25
        plx = self.ObjectParameters['plx']
        
        data = pd.DataFrame(columns=['ra1', 'dec1', 'ra2', 'dec2', 
                                     'ra_ph', 'dec_ph', 'ra_bs', 'dec_bs',
                                     'ra_ss', 'dec_ss', 'ra_nps', 'dec_nps'
                                     'factra', 'factdec', 'ra_bs_plx', 'dec_bs_plx',
                                     'ra_ss_plx', 'dec_ss_plx', 'ra_nps_plx', 'dec_nps_plx',
                                     'w_bs', 'w_ss', 'w_nps'])
        
        data['ra1'], data['dec1'] = self.orbit(params1, times)
        data['ra2'], data['dec2'] = self.orbit(params2, times)
        
        if self.perturbation:
            if isinstance(self.perturbation['value'], (list, tuple, np.ndarray)):
                length = len(times)
                if self.perturbation['value'].shape[1] < length:
                    raise ValueError("The perturbation array is too short. "
                                    "The perturbation array should be greater than DR3 time or higher for DR4 or DR5 simulations. "
                                    f"Expected length: {length}, current: {self.perturbation['value'].shape[1]}.")
                pert_ra, pert_dec = self.perturbation['value'][:,:length]
            else:
                pert_ra, pert_dec = self.perturbation['value'](times)
                
            if self.perturbation['component'] == 1:
                data['ra1'], data['dec1'] = data['ra1'] + pert_ra, data['dec1'] + pert_dec
            else:
                data['ra2'], data['dec2'] = data['ra2'] + pert_ra, data['dec2'] + pert_dec
                
        if self.has_convection:
            
            ra_c, dec_c = self.addConvectionJitter(times)
            
            data['ra1'], data['dec1'] = data['ra1'] + ra_c, data['dec1'] + dec_c
        
        
        if self.ObjectType != 'BH' or self.ObjectType == 'exoplanet':
            a_ph = (self.ObjectParameters['q']/(1+self.ObjectParameters['q'])*np.mean(fdata['r1']) -\
                    1/(1+self.ObjectParameters['q'])*np.mean(fdata['r2']))*self.ObjectParameters['a']
            self.params_ph = {'a': a_ph, 
                       'i': self.ObjectParameters['i'], 
                       'Omega': self.ObjectParameters['Omega'], 
                       'e':self.ObjectParameters['e'],
                       'w':(self.ObjectParameters['w']+180), 
                       'T0': ((self.ObjectParameters['T0']-self.Tref.jd)*u.day).value, 
                       'P':self.ObjectParameters['P']}
        else:
            self.params_ph = params1
            
        # photocenter position 
        data['ra_ph'] = data['ra1']*fdata['r1'].values + data['ra2']*fdata['r2'].values
        data['dec_ph'] = data['dec1']*fdata['r1'].values + data['dec2']*fdata['r2'].values
        
        
        # adding proper motion to the binary system (bs)
        data['ra_bs'] = data['ra_ph'] + pmra*times
        data['dec_bs'] = data['dec_ph'] + pmdec*times

        # proper motion alone to model single star (ss)
        data['ra_ss'] = pmra*times
        data['dec_ss'] = pmdec*times
        
        if self.has_pulsation:
            data['ra_ph_nps'] = (data['ra1']*fdata['r1_nps'] + data['ra2']*fdata['r2_nps'])
            data['dec_ph_nps'] = (data['dec1']*fdata['r1_nps'] + data['dec2']*fdata['r2_nps'])
            data['ra_nps'] = (data['ra1']*fdata['r1_nps'] + data['ra2']*fdata['r2_nps']) + pmra*times
            data['dec_nps'] = (data['dec1']*fdata['r1_nps'] + data['dec2']*fdata['r2_nps']) + pmdec*times
            self.a_ph_min = (self.ObjectParameters['q']/(1+self.ObjectParameters['q'])*np.min(fdata['r1']) -\
                    1/(1+self.ObjectParameters['q'])*np.max(fdata['r2']))*self.ObjectParameters['a']
            self.a_ph_max = (self.ObjectParameters['q']/(1+self.ObjectParameters['q'])*np.max(fdata['r1']) -\
                    1/(1+self.ObjectParameters['q'])*np.min(fdata['r2']))*self.ObjectParameters['a']
        
        # adding projected parallax motion for visualisation
        data['ra_bs_plx'] = data['ra_bs']+plx*factra
        data['dec_bs_plx'] = data['dec_bs']+plx*factdec
        
        data['ra_ss_plx'] = data['ra_ss']+plx*factra
        data['dec_ss_plx'] = data['dec_ss']+plx*factdec
        
        if self.has_pulsation:
            data['ra_nps_plx'] = data['ra_nps']+plx*factra
            data['dec_nps_plx'] = data['dec_nps']+plx*factdec
            
        
        return data
    
    def SimWAL(self, errCCD):
        """
        Simulates along scan positions of the target.
        Parameters
        ----------
        errCCD : Boolean. Default is False.

        Returns
        -------
        Array of along scan positions
        """
        fdata = self.FluxRatio(self.reltimes)
        self.FluxData = fdata
        
        # projecting parallax factors to ra, dec
        factra = -self.plxFactorAL*np.sin(self.scanAngleRAD)+self.plxFactorAC*np.cos(self.scanAngleRAD)
        factdec = self.plxFactorAL*np.cos(self.scanAngleRAD)+self.plxFactorAC*np.sin(self.scanAngleRAD)
        
        data = self.SimGaia(self.reltimes, fdata, factra, factdec)
        self.Data = data
        
        self.w_bs = (self.dec0 + data['dec_bs'])*np.cos(self.scanAngleRAD) \
            + (self.ra0 + data['ra_bs'])*np.sin(self.scanAngleRAD) \
                + self.ObjectParameters['plx']*self.plxFactorAL

        self.w_ss = (self.dec0 + data['dec_ss'])*np.cos(self.scanAngleRAD) \
            + (self.ra0 + data['ra_ss'])*np.sin(self.scanAngleRAD) \
                + self.ObjectParameters['plx']*self.plxFactorAL
        
        if self.has_pulsation:
            self.w_nps = (self.dec0 + data['dec_nps'])*np.cos(self.scanAngleRAD) \
                + (self.ra0 + data['ra_nps'])*np.sin(self.scanAngleRAD) \
                    + self.ObjectParameters['plx']*self.plxFactorAL
        
        if errCCD:
            self.w_bs = self.w_bs + self.errors
        
        return self.w_bs
    
    def orbitTI(self, x, t):
        """
        Orbit function with Thiele-Innes.
        Parameters
        ----------
        x : array_like. P, e, A, F, B, G, T, where AFBG are Thiele-Innes.
        t : array_like.

        Returns
        -------
        RA and DEC arrays of the orbit
        """
        P, e, A, F, B, G, T = x

        M = 2*np.pi/P*(t-T)
        E, cosE, sinE = kepler.kepler(M, e)
        
        X = cosE - e
        Y = np.sqrt(1-e**2)*sinE
        
        delt_ra = B*X + G*Y
        delt_dec = A*X + F*Y

        return np.array([delt_ra, delt_dec])
    
    def residuals(self, x, t, y):
        """
        Calculates residuals.
        Parameters
        ----------
        x : array_like. Orbit parameters P, e, A, F, B, G, T, where AFBG are Thiele-Innes.
        t : array_like. Time.
        y : array_like.

        Returns
        -------
        Array of residuals
        """
        return ((self.orbitTI(x, t)-y)**2).ravel()
    
    def get_dataframe(self, data_dir=None):
        """
        Gives dataframe containing simulated data.
        Parameters
        ----------
        data_dir: str. Default is None.

        Returns
        -------
        Dataframe containing simulated data.
        """
        # resulting datatframe
        sim_astrometry = pd.DataFrame()
        sim_astrometry['transit_id'] = np.arange(1,len(self.timesjd)+1,1) # we don't have the true ones
        sim_astrometry['ccd_id'] = 1 # because we simulate only one ccd
        sim_astrometry['obs_time_tcb'] = self.timesjd
        sim_astrometry['centroid_pos_al'] = self.w_bs
        sim_astrometry['centroid_pos_error_al'] = self.errALCCD(self.ObjectGmag)
        sim_astrometry['parallax_factor_al'] = self.plxFactorAL
        sim_astrometry['scan_pos_angle'] = self.scanAngleDEG # BH3 notebook is made for degrees
        sim_astrometry['outlier_flag'] = 0
        
        if data_dir is not None:
            sim_astrometry.to_csv(data_dir+f'sim{self.ObjectName}_DR{str(self.DataRelease)}.dat', 
                                  sep=' ', header=False, index=False)
        return sim_astrometry
    
    def SimPlot(self, times, fdata=None):
        """
        Produces data to plot.
        Parameters
        ----------
        times : array_like.
        fdata : dictionary containing flux ratios. Default is None.

        Returns
        -------
        Dataframe containing simulated data.
        """
        if fdata is None:
            fdata = self.FluxRatio(times)
        
        factra = -self.plxFactorAL*np.sin(self.scanAngleRAD)+self.plxFactorAC*np.cos(self.scanAngleRAD)
        factdec = self.plxFactorAL*np.cos(self.scanAngleRAD)+self.plxFactorAC*np.sin(self.scanAngleRAD)
        
        factors = np.array([factra, factdec])

        cs = CubicSpline(self.reltimes, factors, axis=1)
        interp = cs(times)
        fitra, fitdec = interp
        
        data = self.SimGaia(times, fdata, fitra, fitdec)
        # self.DataIntep = data
        return data 
    
    def Plot(self, plot_dir=None, Npoints=500, scan_axis=None, scan_length=None, errCCD=False):
        """
        Plots simulated data. Orbit + On-sky.
        Parameters
        ----------
        plot_dir : str. Default is None.
        Npoints : int. Default is 500. Points to interpolate.
        scan_axis : array_like. Default is None. Add scan direction
        scan_length : array_like. Default is None. Length of scan to plot.
        Can be [x,y] where x is for ax1 and y is for ax2. Or just x for both subplots.
        errCCD : Boolean. Default is False. If True add errors to the plot (Not error bars).

        Returns
        -------
        ax1, ax2 of subplots
        """
        
        if self.perturbation:
            if isinstance(self.perturbation['value'], (list, tuple, np.ndarray)):
                perturbation =  self.perturbation
                self.perturbation = None
            perturbation = self.perturbation
        else:
            perturbation =  None
            
        has_convection = self.has_convection
        self.has_convection = False
                
        Period = self.ObjectParameters['P']
        timesOrb = np.linspace(-Period/2, Period/2, Npoints) 
        dataOrb = self.SimPlot(timesOrb)
        
        self.has_convection = has_convection
        
        timesSky = np.linspace(np.min(self.reltimes), np.max(self.reltimes), Npoints) 
        dataSky = self.SimPlot(timesSky)
                
        self.perturbation = perturbation
        
        if self.ObjectType=='cepheid':
            label1 = 'Cepheid'
            label2 = 'Companion'
            lw = 1
        elif self.ObjectType=='binary':
            label1 = 'Star 1'
            label2 = 'Star 2'
            lw = 1
        elif self.ObjectType=='BH':
            label1 = 'Star'
            label2 = 'Black hole'
            lw = 5
        elif self.ObjectType=='exoplanet':
            label1 = 'Host Star'
            label2 = 'Exoplanet'
            lw = 5
        elif self.ObjectType=='AGB':
            label1 = 'AGB'
            label2 = 'Companion'
            lw = 1
        else:
            raise KeyError(f"Unknown type {self.ObjectType}")
            
        fig, axs = plt.subplots(1,2, figsize=(14, 7), constrained_layout=True)
        fig.suptitle(self.ObjectName)
        
        maincolor='black'
        
        ax1, ax2 = axs
        
        ax1.set_title('Orbital motion')
        ax1.plot(dataOrb['ra1'], dataOrb['dec1'], label=label1, color = 'pink', lw = lw, zorder=1)
        ax1.plot(dataOrb['ra2'], dataOrb['dec2'], label=label2, color = 'lightskyblue', zorder=2)
        ax1.plot(dataOrb['ra_ph'], dataOrb['dec_ph'], label='Photocentre', color = maincolor, zorder=3)
        ax1.scatter(self.Data['ra1'], self.Data['dec1'], color = 'pink', zorder=1, s=5)
        ax1.scatter(self.Data['ra2'], self.Data['dec2'], color = 'lightskyblue', zorder=2, s=5)
        ax1.scatter(self.Data['ra_ph'], self.Data['dec_ph'], color = maincolor, zorder=3, s=10)
        ax1.xaxis.set_inverted(True)
        ax1.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax1.set_ylabel(r'$\Delta \delta$ [mas]')
        ax1.legend()
        ax1.set_aspect('equal', adjustable='datalim')
        
        ra_shift1, dec_shift1 = np.mean(dataSky['ra_ss_plx']), np.mean(dataSky['dec_ss_plx'])
        ra_shift2, dec_shift2 = np.mean(dataSky['ra_bs_plx']), np.mean(dataSky['dec_bs_plx'])
        
        ax2.set_title('On-sky (orbital + proper + parallax motions)')
        ax2.plot(dataSky['ra_ss_plx']-ra_shift1, dataSky['dec_ss_plx']-dec_shift1, 
                    label='Single star model', color = 'blueviolet', zorder=1)
        ax2.plot(dataSky['ra_bs_plx']-ra_shift2, dataSky['dec_bs_plx']-dec_shift2, 
                    label='Photocentre of the system', color = maincolor, zorder=2)
        
        ax2.scatter(self.Data['ra_ss_plx']-ra_shift1, self.Data['dec_ss_plx']-dec_shift1, 
                    color = 'blueviolet', zorder=1, s=5)
        ax2.scatter(self.Data['ra_bs_plx']-ra_shift2, self.Data['dec_bs_plx']-dec_shift2, 
                    color = maincolor, zorder=2, s=5)
        ax2.xaxis.set_inverted(True)
        ax2.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax2.set_ylabel(r'$\Delta \delta$ [mas]')
        ax2.legend()
        ax2.set_aspect('equal', adjustable='datalim')
        
        if scan_axis is None or scan_axis=='all' or scan_axis==[1,2]:
            scan_axis=[True, True]
        elif scan_axis==1:
            scan_axis=[True, False]
        elif scan_axis==2:
            scan_axis=[False, True]
        else:
            raise ValueError(f"scan_axis can be 1, 2, [1,2], or 'all', currently {scan_axis}")
        
        ra = np.array([self.Data['ra_ph'], self.Data['ra_bs_plx']-ra_shift2])
        dec = np.array([self.Data['dec_ph'], self.Data['dec_bs_plx']-dec_shift2])
        
        if scan_length is not None:
            if isinstance(scan_length, list):
                if len(scan_length) == 1:
                    scan_length=[scan_length[0], scan_length[0]]
                elif len(scan_length) != 2:
                    raise ValueError("scan_length can not contain more than 2 values.")
            else:
                scan_length=[scan_length, scan_length]
            scan_length = np.array(scan_length)
            
            for axi, ri, di, li in zip(axs[scan_axis], ra[scan_axis], dec[scan_axis], scan_length[scan_axis]):
                for r1, d1, ai in zip(ri, di, self.scanAngleRAD):
                    dx = 0.5 * li * np.sin(ai)
                    dy = 0.5 * li * np.cos(ai)
                    axi.plot([r1 - dx, r1 + dx], [d1 - dy, d1 + dy], color='gray', lw=1, alpha=0.5)
                    
        if errCCD and not self.errCCD:
            raise KeyError('To plot errors (errCCD=True), simulate it with SimBinary first.')
        elif errCCD and self.errCCD:
            
            errx = self.errors * np.sin(self.scanAngleRAD)
            erry = self.errors * np.cos(self.scanAngleRAD)
            
            for axi, ri, di in zip(axs[scan_axis], ra[scan_axis], dec[scan_axis]):
                axi.scatter(ri+errx, di+erry, s = 20, color = 'gray', alpha=0.3, zorder=1.5)
            
        if plot_dir is not None:
            fig.savefig(plot_dir+f'astrometry_gaia_{self.ObjectName}_DR{str(self.DataRelease)}.png', 
                        dpi=300, bbox_inches="tight", transparent=False)
            
        return axs
        
    def PlotCepheid(self, plot_dir= None, Npoints=500):
        """
            Plots simulated data for VIM. Pulsation + Orbit + On-sky.
            Parameters
            ----------
            plot_dir : str. Default is None.
            Npoints : int. Default is 500. Points to interpolate.

            Returns
            -------
            ax1, ax2, ax3 of subplots
        """
        if not self.has_pulsation:
            raise KeyError(f"This plot is only for VIM (cepheid or AGB). The current type is {self.ObjectType}")
        
        
        if self.perturbation:
            if isinstance(self.perturbation['value'], (list, tuple, np.ndarray)):
                perturbation =  self.perturbation
                self.perturbation = None
            perturbation = self.perturbation
        else:
            perturbation =  None
            
        has_convection = self.has_convection
        self.has_convection = False
                
        Period = self.ObjectParameters['P']
        timesOrb = np.linspace(-Period/2, Period/2, Npoints)
        dataOrb = self.SimPlot(timesOrb)
        
        self.has_convection = has_convection
        
        timesSky = np.linspace(np.min(self.reltimes), np.max(self.reltimes), Npoints)
        dataSky = self.SimPlot(timesSky)
                
        self.perturbation = perturbation
        
        
        if self.ObjectType == 'cepheid':
            label1 = 'Cepheid'
        elif self.ObjectType == 'AGB':
            label1 = 'AGB'
        
        label2 = 'Companion'
        
        fig = plt.figure(constrained_layout=True, figsize=(14, 7))
        fig.suptitle(self.ObjectName)
        gs = GridSpec(2, 2, figure=fig, width_ratios=[2, 3])
        
        maincolor='black'
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[:, 1])
        
        # Pulsation
        Ppuls = self.ObjectParameters['Ppuls']
        
        timesPuls = np.linspace(-Ppuls/2, Ppuls/2, 1000)
        dataPuls = self.FluxRatio(timesPuls)
        
        Tplot = timesPuls[np.where(dataPuls['puls'] == np.min(dataPuls['puls']))]
        
        timesPuls = np.linspace(-Ppuls, Ppuls, Npoints)
        dataPuls = self.FluxRatio(timesPuls, Tplot = Tplot)

        ax1.set_title('Photometric variation')
        ax1.plot(timesPuls, dataPuls['puls'], color = 'pink', lw = 3)
        ax1.set_xlabel('Time [day]')
        ax1.set_ylabel('Gmag [mag]')
        ax1.yaxis.set_inverted(True)
        
        # Orbit
        
        Period = self.ObjectParameters['P']
        
        temp = self.params_ph
        temp['a'] = self.a_ph_min
        ra_min, dec_min = self.orbit(temp, timesOrb)
        temp['a'] = self.a_ph_max
        ra_max, dec_max = self.orbit(temp, timesOrb)
        x_poly = np.concatenate([ra_min, ra_max[::-1]])
        y_poly = np.concatenate([dec_min, dec_max[::-1]])
        
        ax2.set_title('Orbital motion')
        ax2.fill(x_poly, y_poly, alpha=0.2, color = maincolor, label = 'VIM zone', lw=0, zorder=2.5)
        ax2.plot(dataOrb['ra1'], dataOrb['dec1'], label=label1, color = 'pink', zorder=1)
        ax2.plot(dataOrb['ra2'], dataOrb['dec2'], label=label2, color = 'lightskyblue', zorder=2)
        ax2.plot(dataOrb['ra_ph_nps'], dataOrb['dec_ph_nps'], label='Mean photocentre', color = maincolor, zorder=3)
        
        ax2.scatter(self.Data['ra1'], self.Data['dec1'], color = 'pink', zorder=1, s=5)
        ax2.scatter(self.Data['ra2'], self.Data['dec2'], color = 'lightskyblue', zorder=2, s=5)
        ax2.scatter(self.Data['ra_ph'], self.Data['dec_ph'], color = maincolor, zorder=3, s=5)
        
        ax2.xaxis.set_inverted(True)
        ax2.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax2.set_ylabel(r'$\Delta \delta$ [mas]')
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.legend()
        
        # Sky
        
        ra_shift1, dec_shift1 = np.mean(dataSky['ra_ss_plx']), np.mean(dataSky['dec_ss_plx'])
        ra_shift2, dec_shift2 = np.mean(dataSky['ra_nps_plx']), np.mean(dataSky['dec_nps_plx'])
        
        f_comp = 10**(-0.4*(self.ObjectParameters['Vcomp']-self.ObjectGmag))
        f_ceph = 10**(-0.4*(dataPuls['puls_ceph']-self.ObjectGmag))
        
        fdataMin = pd.DataFrame(columns=['r1', 'r2'])
        fdataMin['r1'] = [np.min(f_ceph)/(np.min(f_ceph)+f_comp)]
        fdataMin['r2'] = [f_comp/(np.min(f_ceph)+f_comp)]
        
        fdataMax = pd.DataFrame(columns=['r1', 'r2'])
        fdataMax['r1'] = [np.max(f_ceph)/(np.max(f_ceph)+f_comp)]
        fdataMax['r2'] = [f_comp/(np.max(f_ceph)+f_comp)]
        
        self.has_pulsation = False
        dataMin = self.SimPlot(timesSky, fdataMin)
        dataMax = self.SimPlot(timesSky, fdataMax)
        self.has_pulsation = True
        
        ra_min = dataMin['ra_bs_plx']-ra_shift2
        ra_max = dataMax['ra_bs_plx']-ra_shift2
        dec_min = dataMin['dec_bs_plx']-dec_shift2
        dec_max = dataMax['dec_bs_plx']-dec_shift2
        
        label_sys='Binary system'
        if self.ObjectType == 'AGB':
            label_sys = 'Binary system with AGB'
        
        ax3.set_title('On-sky (orbital + proper + parallax motions)')
        ax3.plot(dataSky['ra_ss_plx']-ra_shift1, dataSky['dec_ss_plx']-dec_shift1, 
                    label='Single star model', color = 'plum', zorder=1)
        ax3.plot(dataSky['ra_nps_plx']-ra_shift2, dataSky['dec_nps_plx']-dec_shift2, 
                 label=label_sys, color='darkviolet', zorder=2, lw=2)
        # ax3.plot(dataSky['ra_bs_plx']-ra_shift2, dataSky['dec_bs_plx']-dec_shift2, 
        #             label='Photocentre of the system', color = 'black', alpha =0.1)      
        
        ax3.fill(
            [ra_min[0], ra_min[1], ra_max[1], ra_max[0]],
            [dec_min[0], dec_min[1], dec_max[1], dec_max[0]],
            color = '#D6D6D6', lw=1, zorder=0.5, label = 'VIM zone')
        
        for i in range(len(dataMin['dec_bs_plx'])-1):
            ax3.fill(
                [ra_min[i], ra_min[i+1], ra_max[i+1], ra_max[i]],
                [dec_min[i], dec_min[i+1], dec_max[i+1], dec_max[i]],
                color = '#D6D6D6', lw=1, zorder=0.5)
        
        ax3.scatter(self.Data['ra_ss_plx']-ra_shift1, self.Data['dec_ss_plx']-dec_shift1, 
                    color = 'plum', zorder=1, s=5)
        ax3.scatter(self.Data['ra_nps_plx']-ra_shift2, self.Data['dec_nps_plx']-dec_shift2, 
                 color='darkviolet', zorder=2, s=5)
        ax3.scatter(self.Data['ra_bs_plx']-ra_shift2, self.Data['dec_bs_plx']-dec_shift2, 
                    color = maincolor, zorder=3, s=5, label = 'VIM')
        
        ax3.xaxis.set_inverted(True)
        ax3.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax3.set_ylabel(r'$\Delta \delta$ [mas]')
        ax3.set_aspect('equal', adjustable='datalim')
        ax3.legend()
        
        if plot_dir is not None:
            fig.savefig(plot_dir+f'astrometry_gaia_cepheid_{self.ObjectName}_DR{str(self.DataRelease)}.png', 
                        dpi=300, bbox_inches="tight")
        return [ax1, ax2, ax3]
    
    def PlotCepheidRow(self, plot_dir= None, Npoints=500):
        """
            Plots simulated data for VIM. Pulsation + Orbit + On-sky.
            Parameters
            ----------
            plot_dir : str. Default is None.
            Npoints : int. Default is 500. Points to interpolate.

            Returns
            -------
            ax1, ax2, ax3 of subplots
        """
        if not self.has_pulsation:
            raise KeyError(f"This plot is only for VIM (cepheid or mira). The current type is {self.ObjectType}")
        
        if self.perturbation:
            if isinstance(self.perturbation['value'], (list, tuple, np.ndarray)):
                perturbation =  self.perturbation
                self.perturbation = None
            perturbation = self.perturbation
        else:
            perturbation =  None
            
        has_convection = self.has_convection
        self.has_convection = False
                
        Period = self.ObjectParameters['P']
        timesOrb = np.linspace(-Period/2, Period/2, Npoints)
        dataOrb = self.SimPlot(timesOrb)
        
        self.has_convection = has_convection
        
        timesSky = np.linspace(np.min(self.reltimes), np.max(self.reltimes), Npoints)
        dataSky = self.SimPlot(timesSky)
                
        self.perturbation = perturbation
        
        label1 = 'Cepheid'
        label2 = 'Companion'
        fig, axs = plt.subplots(1,3, figsize=(15, 5), constrained_layout=True)
        fig.suptitle(self.ObjectName)
        
        ax1, ax2, ax3 = axs
        
        # Pulsation
        Ppuls = self.ObjectParameters['Ppuls']
        
        timesPuls = np.linspace(-Ppuls/2, Ppuls/2, 1000)
        dataPuls = self.FluxRatio(timesPuls)
        
        Tplot = timesPuls[np.where(dataPuls['puls'] == np.min(dataPuls['puls']))]
        
        timesPuls = np.linspace(-Ppuls, Ppuls, Npoints)
        dataPuls = self.FluxRatio(timesPuls, Tplot = Tplot)

        ax1.set_title('Photometric variation')
        ax1.plot(timesPuls, dataPuls['puls'], color = 'pink', lw = 3)
        ax1.set_xlabel('Time [day]')
        ax1.set_ylabel('Gmag [mag]')
        ax1.yaxis.set_inverted(True)
        
        # Orbit
        
        Period = self.ObjectParameters['P']
        
        temp = self.params_ph
        temp['a'] = self.a_ph_min
        ra_min, dec_min = self.orbit(temp, timesOrb)
        temp['a'] = self.a_ph_max
        ra_max, dec_max = self.orbit(temp, timesOrb)
        x_poly = np.concatenate([ra_min, ra_max[::-1]])
        y_poly = np.concatenate([dec_min, dec_max[::-1]])
        
        ax2.set_title('Orbital motion')
        ax2.fill(x_poly, y_poly, alpha=0.2, color = 'black', label = 'VIM zone', lw=0, zorder=2.5)
        ax2.plot(dataOrb['ra1'], dataOrb['dec1'], label=label1, color = 'pink', zorder=1)
        ax2.plot(dataOrb['ra2'], dataOrb['dec2'], label=label2, color = 'lightskyblue', zorder=2)
        # ax2.plot(dataOrb['ra_ph_nps'], dataOrb['dec_ph_nps'], label='Mean photocentre', color = 'black', zorder=3)
        
        ax2.scatter(self.Data['ra1'], self.Data['dec1'], color = 'pink', zorder=1, s=5)
        ax2.scatter(self.Data['ra2'], self.Data['dec2'], color = 'lightskyblue', zorder=2, s=5)
        ax2.scatter(self.Data['ra_ph'], self.Data['dec_ph'], color = 'black', zorder=3, s=5)
        
        ax2.xaxis.set_inverted(True)
        ax2.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax2.set_ylabel(r'$\Delta \delta$ [mas]')
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.legend()
        
        ax2.scatter(0,0, marker='x', color='black', s=15, alpha=0.7)
        
        # Sky
        
        ra_shift1, dec_shift1 = np.mean(dataSky['ra_ss_plx']), np.mean(dataSky['dec_ss_plx'])
        ra_shift2, dec_shift2 = np.mean(dataSky['ra_nps_plx']), np.mean(dataSky['dec_nps_plx'])
        
        f_comp = 10**(-0.4*(self.ObjectParameters['Vcomp']-self.ObjectGmag))
        f_ceph = 10**(-0.4*(dataPuls['puls_ceph']-self.ObjectGmag))
        
        fdataMin = pd.DataFrame(columns=['r1', 'r2'])
        fdataMin['r1'] = [np.min(f_ceph)/(np.min(f_ceph)+f_comp)]
        fdataMin['r2'] = [f_comp/(np.min(f_ceph)+f_comp)]
        
        fdataMax = pd.DataFrame(columns=['r1', 'r2'])
        fdataMax['r1'] = [np.max(f_ceph)/(np.max(f_ceph)+f_comp)]
        fdataMax['r2'] = [f_comp/(np.max(f_ceph)+f_comp)]
        
        self.has_pulsation = False
        dataMin = self.SimPlot(timesSky, fdataMin)
        dataMax = self.SimPlot(timesSky, fdataMax)
        self.has_pulsation = True
        
        ra_min = dataMin['ra_bs_plx']-ra_shift2
        ra_max = dataMax['ra_bs_plx']-ra_shift2
        dec_min = dataMin['dec_bs_plx']-dec_shift2
        dec_max = dataMax['dec_bs_plx']-dec_shift2
        
        ax3.set_title('On-sky (orbital + proper + parallax motions)')
        ax3.plot(dataSky['ra_ss_plx']-ra_shift1, dataSky['dec_ss_plx']-dec_shift1, 
                    label='Single star model', color = 'plum', zorder=1)
        ax3.plot(dataSky['ra_nps_plx']-ra_shift2, dataSky['dec_nps_plx']-dec_shift2, 
                 label='Binary system', color='darkviolet', zorder=2, lw=2)
        # ax3.plot(dataSky['ra_bs_plx']-ra_shift2, dataSky['dec_bs_plx']-dec_shift2, 
        #             label='Photocentre of the system', color = 'black', alpha =0.1)      
        
        ax3.fill(
            [ra_min[0], ra_min[1], ra_max[1], ra_max[0]],
            [dec_min[0], dec_min[1], dec_max[1], dec_max[0]],
            color = '#D6D6D6', lw=1, zorder=0.5, label = 'VIM zone')
        
        for i in range(len(dataMin['dec_bs_plx'])-1):
            ax3.fill(
                [ra_min[i], ra_min[i+1], ra_max[i+1], ra_max[i]],
                [dec_min[i], dec_min[i+1], dec_max[i+1], dec_max[i]],
                color = '#D6D6D6', lw=1, zorder=0.5)
        
        ax3.scatter(self.Data['ra_ss_plx']-ra_shift1, self.Data['dec_ss_plx']-dec_shift1, 
                    color = 'plum', zorder=1, s=5)
        ax3.scatter(self.Data['ra_nps_plx']-ra_shift2, self.Data['dec_nps_plx']-dec_shift2, 
                 color='darkviolet', zorder=2, s=5)
        ax3.scatter(self.Data['ra_bs_plx']-ra_shift2, self.Data['dec_bs_plx']-dec_shift2, 
                    color = 'black', zorder=3, s=5, label = 'VIM')
        
        ax3.xaxis.set_inverted(True)
        ax3.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax3.set_ylabel(r'$\Delta \delta$ [mas]')
        ax3.set_aspect('equal', adjustable='datalim')
        ax3.legend()
        
        if plot_dir is not None:
            fig.savefig(plot_dir+f'astrometry_gaia_cepheid_{self.ObjectName}_DR{str(self.DataRelease)}.png', 
                        dpi=300, bbox_inches="tight")
        return [ax1, ax2, ax3]
    
    
    
    def PlotSSfit(self, plot_dir=None):
        """
        Plots fitted Single-Star model to the object.
        Parameters
        ----------
        plot_dir : str. optional. Default = None.

        Returns
        -------
        ax1, ax2 of subplots
        """
        
        factra = -self.plxFactorAL*np.sin(self.scanAngleRAD)+self.plxFactorAC*np.cos(self.scanAngleRAD)
        factdec = self.plxFactorAL*np.cos(self.scanAngleRAD)+self.plxFactorAC*np.sin(self.scanAngleRAD)
        
        p_fit, errors, w_fit = self.fitSS(self.w_bs)
        r0, pmr, d0, pmd, plx = p_fit
        ra = r0 + pmr*self.reltimes + plx*factra
        dec = d0 + pmd*self.reltimes + plx*factdec
        
        werr = np.array(len(w_fit)*[self.errALCCD(self.ObjectGmag)])
        
        ra_shift1, dec_shift1 = np.mean(self.Data['ra_bs_plx']), np.mean(self.Data['dec_bs_plx'])
        ra_shift2, dec_shift2 = np.mean(ra), np.mean(dec)
        
        fig, axs = plt.subplots(2,2, figsize=(14, 7), constrained_layout=True,
                                height_ratios=[4, 1])
        ax1, ax2, ax3, ax4 = axs.ravel()
        fig.suptitle(self.ObjectName, fontsize=16)
        
        ax1.set_title('On sky (orbit + proper + parallax motions)')
        ax1.plot(self.Data['ra_bs_plx']-ra_shift1, self.Data['dec_bs_plx']-dec_shift1, label = 'Data points', marker='.', color = 'coral')
        ax1.plot(ra-ra_shift2, dec-dec_shift2, label = 'Fit', marker='.', color = 'black')
        ax1.set_xlabel(r'$\Delta \alpha cos(\delta)$ [mas]')
        ax1.set_ylabel(r'$\Delta \delta$ [mas]')
        ax1.xaxis.set_inverted(True)
        ax1.legend()
        
        ax3.set_title('Along scan positions')
        ax2.scatter(self.reltimes, self.w_bs, label = 'Data points', color = 'coral', s=50, zorder=1)
        ax2.errorbar(self.reltimes, self.w_bs, yerr = werr, color = 'coral', ls='', zorder=1)
        ax2.scatter(self.reltimes, w_fit, label = 'Fit', color = 'black', s=20, zorder=2)
        ax2.set_ylabel('AL positions [mas]', fontsize=14)
        ax2.legend()
        
        res2d = np.sqrt((self.Data['ra_bs_plx']-ra)**2 + (self.Data['dec_bs_plx']-dec)**2)
        ax3.scatter(self.reltimes, res2d, s=10, color = 'black')
        ax3.set_ylabel('Sky residuals [mas]', fontsize=12)

        res1d = self.w_bs-w_fit
        ax4.scatter(self.reltimes, res1d, s=10, color = 'black')
        ax4.set_ylabel('AL residuals [mas]', fontsize=12)
        
        if plot_dir is not None:
            fig.savefig(plot_dir+f'fitSS_{self.ObjectName}_DR{str(self.DataRelease)}.png', 
                        dpi=300, bbox_inches="tight")
            
        return axs
    def fitSS(self, w_bs=None):
        """
        Fits Single-Star model to given array.
        Parameters
        ----------
        w_bs : array-like. Along scan positions.

        Returns
        -------
        fitter parameters, their errors, and fitted along-scan positions array.
        """
        
        if w_bs is None:
            w_bs = self.w_bs
        
        mA = np.array([
            np.sin(self.scanAngleRAD),                # alpha0
            self.reltimes*np.sin(self.scanAngleRAD),  # pmra
            np.cos(self.scanAngleRAD),                # delta0
            self.reltimes*np.cos(self.scanAngleRAD),  # pmdec
            self.plxFactorAL                          # parallax
            ]).T
        werr = np.array(len(w_bs)*[self.errALCCD(self.ObjectGmag)])
        Cinv = np.diag(1/werr**2)
        p_fit = np.linalg.solve(mA.T @ Cinv @ mA, mA.T @ Cinv @ w_bs)
        w_fit = mA @ p_fit
        
        chi2r = np.sum(((w_bs-w_fit)/werr)**2)/(len(w_fit)-5)
        print('chi2r', np.round(chi2r, 3))
        
        F = mA.T @ Cinv @ mA
        Cov_p = np.linalg.inv(F)
        errors = np.sqrt(np.diag(Cov_p))
        errors = errors * chi2r
        
        labels = ['a0', 'pmra', 'd0', 'pmdec', 'plx']
        
        for p, e, l in zip(p_fit, errors, labels):
            
            print(l, np.round(p,4), '\u00B1', np.round(e,4))
        
        return p_fit, errors, w_fit
    
    def fitVIMF(self, w_bs=None):
        """
        Fits Variability-Induced Mover Fixed model to given array.
        Parameters
        ----------
        w_bs : array-like. Along scan positions.
    
        Returns
        -------
        fitter parameters, their errors, and fitted along-scan positions array.
        """
        
        if w_bs is None:
            w_bs = self.w_bs
            
        flux = self.FluxData['f_tot']
        fref = np.mean(flux)
        werr = np.array(len(w_bs)*[self.errALCCD(self.ObjectGmag)])
        
        mA = np.array([
            np.sin(self.scanAngleRAD),                # alpha0
            self.reltimes*np.sin(self.scanAngleRAD),  # pmra
            np.cos(self.scanAngleRAD),                # delta0
            self.reltimes*np.cos(self.scanAngleRAD),  # pmdec
            self.plxFactorAL,                         # parallax
            (fref/flux-1)*np.sin(self.scanAngleRAD),  # Da
            (fref/flux-1)*np.cos(self.scanAngleRAD),  # Dd
            ]).T
        Cinv = np.diag(1/werr**2)
        mu = np.linalg.solve(mA.T @ Cinv @ mA, mA.T @ Cinv @ w_bs)
        w_fit = np.dot(mA, mu)
        return w_fit, mu
    
    def fitVIML(self, w_bs=None):
        """
        Fits Variability-Induced Mover Linear model to given array.
        Parameters
        ----------
        w_bs : array-like. Along scan positions.
    
        Returns
        -------
        fitter parameters, their errors, and fitted along-scan positions array.
        """
        
        if w_bs is None:
            w_bs = self.w_bs
            
        flux = self.FluxData['f_tot']
        fref = np.mean(flux)
        werr = np.array(len(w_bs)*[self.errALCCD(self.ObjectGmag)])
        
        mA = np.array([
            np.sin(self.scanAngleRAD),                             # alpha0
            self.reltimes*np.sin(self.scanAngleRAD),               # pmra
            np.cos(self.scanAngleRAD),                             # delta0
            self.reltimes*np.cos(self.scanAngleRAD),               # pmdec
            self.plxFactorAL,                                       # parallax
            (fref/flux-1)*np.sin(self.scanAngleRAD),               # Da
            (fref/flux-1)*self.reltimes*np.sin(self.scanAngleRAD), # Dat
            (fref/flux-1)*np.cos(self.scanAngleRAD),               # Dd
            (fref/flux-1)*self.reltimes*np.cos(self.scanAngleRAD)  # Ddt
            ]).T
        Cinv = np.diag(1/werr**2)
        mu = np.linalg.solve(mA.T @ Cinv @ mA, mA.T @ Cinv @ w_bs)
        w_fit = np.dot(mA, mu)
        return w_fit, mu

    def VIMA(self, parameters):
        """
        Defines the VIMA model
        Parameters
        ----------
        parameters : parameters of the VIMA model
    
        Returns
        -------
        fitter parameters, their errors, and fitted along-scan positions array.
        """
            
        flux = self.FluxData['f_tot']
        fref = np.mean(flux)
        
    
        alpha0, pmra, delta0, pmdec, parallax, Da, Dd, Dat, Ddt, k, s = parameters
    
        dDat = -k * Dd
        dDdt =  k * Da
    
        dpmra  = s * dDat
        dpmdec = s * dDdt
    
        factor = (fref/flux - 1)
    
        w_fit = (
            alpha0*np.sin(self.scanAngleRAD)                       # a0
            + pmra*self.reltimes*np.sin(self.scanAngleRAD)         # pmra
            + 0.5*dpmra*self.reltimes**2*np.sin(self.scanAngleRAD) # proper acceleration ra
            + delta0*np.cos(self.scanAngleRAD)                     # d0
            + pmdec*self.reltimes*np.cos(self.scanAngleRAD)        # pmdec
            + 0.5*dpmdec*self.reltimes**2*np.cos(self.scanAngleRAD)# proper acceleration dec
            + parallax*self.plxFactorAL                            # plx
            + factor*(
                Da*np.sin(self.scanAngleRAD)                          # Da
                + Dat*self.reltimes*np.sin(self.scanAngleRAD)         # Da'
                + 0.5*dDat*self.reltimes**2*np.sin(self.scanAngleRAD) # Da"
                + Dd*np.cos(self.scanAngleRAD)                        # Dd
                + Ddt*self.reltimes*np.cos(self.scanAngleRAD)         # Dd'
                + 0.5*dDdt*self.reltimes**2*np.cos(self.scanAngleRAD) # Dd"
            )
        )
    
        return w_fit
    
    def fitVIMA(self, parameters0, w_bs=None):
        """
        Fits Variability-Induced Mover Accelerated model to given array.
        Parameters
        ----------
        parameters0 : initial guess
        w_bs : array-like. Along scan positions.
    
        Returns
        -------
        fitter parameters, their errors, and fitted along-scan positions array.
        """
        
        if w_bs is None:
            w_bs = self.w_bs
            
        werr = np.array(len(w_bs)*[self.errALCCD(self.ObjectGmag)])
        
        def residulas(params):
            w_fit = self.VIMA(params)
            return (w_bs - w_fit) / werr
        
        res_lsq = least_squares(residulas, parameters0)
        w_fitA = self.VIMA(res_lsq.x)
        
        a0A, pmraA, dA, pmdecA, plxA, DaA, DdA, DatA, DdtA, kA, sA = res_lsq.x

        dDatA = -kA * DdA
        dDdtA =  kA * DaA

        dpmraA  = sA * dDatA
        dpmdecA = sA * dDdtA

        p_fitA = [a0A, pmraA, dA, pmdecA, plxA, DaA, DdA, DatA, DdtA, dDatA, dDdtA, dpmraA, dpmdecA]
        
        return w_fitA, p_fitA
        