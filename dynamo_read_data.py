# Read functions for BOWTIE various datasets.
# 
# Soundings - full time series 
# 
# DSHIP ship data
# 
# Radiometer
# 
# Sun photometer
# 
# 
# James Ruppert
# 18 Sept 2024

import numpy as np
import xarray as xr
import subprocess
from thermo_functions import *
import metpy.calc as mpcalc
from metpy.units import units


# data_main = "/ourdisk/hpc/radclouds/auto_archive_notyet/tape_2copies/macsyrett/dynamo-interp/"
data_main = "/Users/jamesruppert/OneDrive\ -\ University\ of\ Oklahoma/code/data/dynamo/"

#############################################
### Sounding data
#############################################

def read_dynamo_soundings():

    #### Main variable read loop

    def read_soundings(files):

        soundings = {}

        for ifile in range(len(files)):

            isnd_file = snd_files[ifile].strip()
            sndfile = xr.open_dataset(isnd_file, engine='netcdf4')
            nz = len(np.squeeze(sndfile['level'].data))
            times = np.squeeze(sndfile['release_time'].data) # seconds since 2011-01-01 00:00:00 UTC
            site_name = sndfile.Site_Name
            sndfile.close()
            nt = len(times)

            # Read in sounding data
            p    = np.squeeze(sndfile['p'].data)*1e2 # Pa
            hght = np.squeeze(sndfile['alt'].data) # m
            tmpk = np.squeeze(sndfile['T'].data)+273.15 # K
            # Convert dew point to relative humidity
            # def get_sh_rh(ds):
            td    = np.squeeze(sndfile['Td'].data)+273.15 # K
            p_mp  = p * units.pascal
            td_mp = td * units.kelvin
            sh    = mpcalc.specific_humidity_from_dewpoint(p_mp, td_mp, phase='liquid')
            mr    = sh2mixr(sh.magnitude)   # kg/kg
            rh    = calc_relh(mr,p,tmpk,ice=True) # %
            wspd  = np.squeeze(sndfile['wind_spd'].data) # deg
            wdir  = np.squeeze(sndfile['wind_dir'].data) # deg
            u, v  = mpcalc.wind_components(wspd*units('m/s'), wdir*units.deg) # m/s
            sndfile.close()
            qcf_p = np.squeeze(sndfile['qcf_p'].data) # qc flag
            where_bad = np.where(qcf_p != 1)
            p[where_bad] = np.nan
            hght[where_bad] = np.nan
            tmpk[where_bad] = np.nan
            rh[where_bad] = np.nan
            mr[where_bad] = np.nan
            u[where_bad] = np.nan
            v[where_bad] = np.nan
            wdir[where_bad] = np.nan
            # Find the first level below 0C
            hght_0c = np.full(nt, np.nan)
            for it in range(nt):
                if np.any(tmpk[it,:] <= 273.15):
                    hght_0c[it] = hght[it, np.where(tmpk[it,:] <= 273.15)[0][0] ]
                else:
                    hght_0c[it] = np.nan
            soundings[site_name] = {
                'site_lon':np.ma.masked_invalid(sndfile['site_lon'].data),
                'site_lat':np.ma.masked_invalid(sndfile['site_lat'].data),
                'hght':np.ma.masked_invalid(hght),
                'hght_0c':np.ma.masked_invalid(hght_0c),
                'p': np.ma.masked_invalid(p),
                'tmpk': np.ma.masked_invalid(tmpk),
                'rh': np.ma.masked_invalid(rh),
                'mr': np.ma.masked_invalid(mr),
                'u': np.ma.masked_invalid(u.magnitude),
                'v': np.ma.masked_invalid(v.magnitude),
                'wdir': np.ma.masked_invalid(wdir),
            }

        return soundings

    #### Call the functions

    # Read list of sounding files
    process = subprocess.Popen(['ls --color=none '+data_main+'*nc'],shell=True,
        stdout=subprocess.PIPE,universal_newlines=True)
    snd_files = process.stdout.readlines()

    # Provides sounding dataset as a dictionary
    soundings = read_soundings(snd_files)

    return soundings

#############################################