#!/usr/bin/env python3
# Python 2/3 compatibility
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')
import os,sys
import numpy as np
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from photutils import SkyCircularAperture
from photutils import CircularAnnulus,aperture_photometry

def is_number(num):
    try:
        num = float(num)
    except ValueError:
        return(False)
    return(True)

def parse_coord(ra, dec):
    if (not (is_number(ra) and is_number(dec)) and
        (':' not in ra and ':' not in dec)):
        return(None)

    if (':' in ra and ':' in dec):
        # Input RA/DEC are sexagesimal
        unit = (u.hourangle, u.deg)
    else:
        unit = (u.deg, u.deg)

    try:
        coord = SkyCoord(ra, dec, frame='icrs', unit=unit)
        return(coord)
    except ValueError:
        return(None)

def add_options(parser=None, usage=None):
    import argparse
    if parser == None:
        parser = argparse.ArgumentParser(usage=usage,conflict_handler='resolve')

    parser.add_argument('filename', type=str,
        help='Right ascension to reduce the HST images')
    parser.add_argument('ra', type=str,
        help='Right ascension to reduce the HST images')
    parser.add_argument('dec', type=str,
        help='Right ascension to reduce the HST images')

    parser.add_argument('--radius', default=3.0, type=float,
        help='Radius of the source aperture in arcseconds.')

    return(parser)

parser = add_options()
opt = parser.parse_args()

filename = opt.filename
if not os.path.exists(filename):
    print('ERROR: file {0} does not exist!'.format(filename))
    print('Exiting...')
    sys.exit()

coord = parse_coord(opt.ra, opt.dec)
if not coord:
    print('ERROR: could not parse ra={0}, dec={0} into a coordinate')
    print('Exiting...')
    sys.exit()

def get_photometry(file, coord, radius=3, significant_figures=4):
    hdu = fits.open(file)

    use_idx=0
    try:
        w = wcs.WCS(hdu[use_idx].header)
    except:
        print('ERROR: file {0} does not have WCS.'.format(file))
        print('Exiting...')
        sys.exit()

    header = hdu[use_idx].header

    if 'CD1_1' in header.keys() and 'CD1_2' in header.keys():
        pscale = np.sqrt(float(header['CD1_1'])**2+float(header['CD1_2'])**2)
        pscale = pscale * 3600.
    elif 'CDELT1' in header.keys():
        pscale = float(header['CDELT1']) * 3600.0
    else:
        print('ERROR: could not parse WCS keywords for {0}'.format(file))
        print('Exiting...')
        sys.exit()

    x,y = wcs.utils.skycoord_to_pixel(coord, w, origin=1, mode='wcs')

    if x < 0 or x > header['NAXIS1'] or y < 0 or y > header['NAXIS2']:
        print(x,y,'not in',file)
        print('Exiting...')
        sys.exit()

    # Construct apertures for photometry and background
    aperture = SkyCircularAperture(coord, radius * u.arcsec)
    phot_table = aperture_photometry(hdu[use_idx].data, aperture,
        wcs=WCS(hdu[use_idx].header), method='exact')
    phot = phot_table['aperture_sum'][0]

    pix_aperture = aperture.to_pixel(w)

    # Now get background estimation
    # As default background aperture use an annulus with inner radius 2*radius
    # and outer radius 4*radius
    background = CircularAnnulus((x,y), r_in = 2 * radius/pscale,
        r_out = 4 * radius/pscale)
    backmask = background.to_mask(method = 'center')
    if isinstance(backmask, list):
        backmask = backmask[0]

    backdata = backmask.multiply(hdu[use_idx].data).flatten()
    mask = backmask.data.flatten()!=0.0
    backdata = backdata[mask]

    # Get sigma-clipped median
    for i in np.arange(10):

        median = np.median(backdata)
        mean = np.mean(backdata)
        std = np.std(backdata)

        mask = np.abs(backdata-median) < 3.0 * std
        backdata = backdata[mask]

    # Rescale the background to the area of the circular aperture
    back   = np.pi * pix_aperture.r**2 * mean
    counts = phot  - back

    backerr = np.std(backdata)

    # Metadata
    h = hdu[use_idx].header

    # Instrumental magnitude
    mag = -2.5 * np.log10(counts)

    # Statistical uncertainty on instrumental magnitude
    #merr = 1.086*np.sqrt(1/np.abs(counts) + 1/np.abs(back))
    merr = 1.086 * np.sqrt(np.pi * pix_aperture.r**2 * backerr**2)/counts

    zpt=0.0
    zerr=0.0
    if 'ZPTMAG' in hdu[use_idx].header.keys():
        zpt=hdu[use_idx].header['ZPTMAG']
        print('zeropoint:',zpt)
    elif 'MAGZERO' in hdu[use_idx].header.keys():
        zpt=hdu[use_idx].header['MAGZERO']
        print('zeropoint:',zpt)
    elif 'PHOTZP' in hdu[use_idx].header.keys():
        zpt=hdu[use_idx].header['PHOTZP']
        print('zeropoint:',zpt)
    elif 'FPA.ZP' in hdu[use_idx].header.keys():
        zpt=hdu[use_idx].header['FPA.ZP']
        zpt+=2.5*np.log10(hdu[use_idx].header['EXPTIME'])
        print('zeropoint:',zpt)
    if 'ZPTMUCER' in hdu[use_idx].header.keys():
        zerr=hdu[use_idx].header['ZPTMUCER']
        print('zpterr:',hdu[use_idx].header['ZPTMUCER'])

    print('Counts, phot, background:',counts,phot,back)
    if np.isnan(mag) or np.isnan(merr):
        return(np.NaN, np.NaN)
    else:
        fmt='%2.{0}f'.format(significant_figures)
        mag=mag+zpt ; magerr=np.sqrt(merr**2+zerr**2)
        mag=float(fmt%mag)
        magerr=float(fmt%magerr)
        return(mag, magerr)


magnitude, error = get_photometry(filename, coord, radius=opt.radius)
print('Got {0}+/-{1} at {2}, {3} for {4}'.format(magnitude, error,
    coord.ra.degree, coord.dec.degree, filename))
