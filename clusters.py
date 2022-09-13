#!/usr/bin/env python3
# Python 2/3 compatibility
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np, os, sys, warnings, glob, copy

binned=False
mask_best_detection=True

def is_number(num):
    try:
        num = float(num)
    except ValueError:
        return(False)
    return(True)

def parse_coord(ra, dec):
    if (not (is_number(ra) and is_number(dec)) and
        (':' not in str(ra) and ':' not in str(dec))):
        return(None)

    if (':' in str(ra) and ':' in str(dec)):
        # Input RA/DEC are sexagesimal
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        coord = SkyCoord(ra, dec, frame='icrs', unit=unit)
        return(coord)
    except ValueError:
        return(None)


def transpose(data):
    return(list(map(list, zip(*data))))

def import_cluster_file(filename):
        idx=0
        table_count=0
        filedata=[]
        photdata=[]
        canddata=[]
        phottabs={}
        with open(filename,'r') as f:
            for line in f:
                if line.startswith('#'):
                    if table_count==0:
                        filehead=line.replace('#','').split()
                    elif table_count==1:
                        candhead=line.replace('#','').split()
                    else:
                        phothead=line.replace('#','').split()
                    if photdata:
                        key=str(photdata[0][0].strip())
                        if key in phottabs.keys():
                            currdata = phottabs[key]
                            currdata.extend(photdata)
                            phottabs[key] = currdata
                        else:
                            phottabs[key]=photdata
                        photdata=[]
                    table_count += 1
                    continue
                if table_count==1:
                    filedata.append(line.split())
                elif table_count==2:
                    canddata.append(line.split())
                else:
                    photdata.append(line.split())
            if photdata:
                phot_key=str(photdata[0][0].strip())
                if phot_key in phottabs.keys():
                    phottabs[phot_key]+=photdata
                else:
                    phottabs[phot_key]=photdata

        filetable = Table(transpose(filedata), names=filehead)
        candtable = Table(transpose(canddata), names=candhead)
        for key in phottabs.keys():
            phottabs[key] = Table(transpose(phottabs[key]), names=phothead)
        return(filetable, candtable, phottabs)

def get_all_tables(obj):
    files = None
    candidates = None
    photometry = []
    for file in glob.glob(obj+'*.clusters'):
        filetable, candtable, phottabs = import_cluster_file(file)
        if not files:
            files = Table(filetable)
        else:
            files = vstack([files, filetable])
        if not candidates:
            candidates = Table(candtable)
        else:
            candidates = vstack([candidates, candtable])
        photometry.append(phottabs)

    return(files, candidates, photometry)

def get_cand_id(coord, candidates):

    cand_coords = SkyCoord(candidates['RAaverage'],
        candidates['DECaverage'], unit=(u.hour, u.deg))
    sep = coord.separation(cand_coords)
    mask=sep<2.0*u.arcsec

    if len(candidates[mask])==0:
        return(None)
    else:
        idx = np.argmin(sep)
        return(candidates[idx]['ID'])


def parse_photometry(cand_ra, cand_dec, objname):

    cand_coord = parse_coord(cand_ra, cand_dec)
    files, candidates, photometry = get_all_tables(objname)

    cand_id = get_cand_id(cand_coord, candidates)

    if not cand_id: return(None)

    outdata = Table([[0.],[''],[0.],[0.],['X'*20],[0.],[0.]],
        names=('MJD','FILTER','MAG','MAGERR','SOURCE','FLUX',
            'FLUXERR')).copy()[:0]

    mask=photometry[0][cand_id]['type']=='0x00000011'
    photometry[0][cand_id]=photometry[0][cand_id][mask]

    for row in photometry[0][cand_id]:
        mask = files['cmpfile']==row['cmpfile']
        zpt = files[mask][0]['ZPTMAGAV']
        mjd = files[mask][0]['MJD']
        phot = files[mask][0]['photcode']

        filt=''
        if phot.endswith('02'): filt='B'
        if phot.endswith('03'): filt='V'
        if phot.endswith('12'): filt='u'
        if phot.endswith('13'): filt='g'
        if phot.endswith('14'): filt='r'
        if phot.endswith('15'): filt='i'
        if phot.endswith('16'): filt='z'

        source=''
        if '94' in str(phot): source='LULIN'
        if '4a' in str(phot): source='LCOGT'

        if float(row['dM'])>0.3:
            mag = float(-2.5*np.log10(10*float(row['dflux'])))+float(zpt)
            magerr = 0.0
        else:
            mag = float(row['M'])+float(zpt)
            magerr = float(row['dM'])+0.01

        outdata.add_row([float(mjd), filt, mag, magerr, source,
            float(row['flux']),float(row['dflux'])])

    if mask_best_detection:
        newtable = outdata.copy()[:0]
        for mjd in np.unique(outdata['MJD']):
            mask = outdata['MJD']==mjd
            if len(outdata[mask])==1:
                newtable.add_row(outdata[mask][0])
            else:
                # Add row with smallest FLUXERR
                subtable = outdata[mask]
                subtable.sort('FLUXERR')
                newtable.add_row(subtable[0])

        outdata = copy.copy(newtable)

    return(outdata)
