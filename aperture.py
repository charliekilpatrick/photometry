#!/usr/bin/env python3
# Python 2/3 compatibility
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')
import os
import sys
import copy
import numpy as np
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from scipy.optimize import curve_fit
from photutils import SkyCircularAperture,CircularAperture
from photutils import CircularAnnulus,aperture_photometry

import matplotlib.pyplot as plt

def complex_background(data, x0, y0, radius_inner=4, radius_outer=16, order=1,
    grow=4, test=False):

    orig_data = copy.copy(data)

    data_shape = data.shape
    print('Shape',data_shape)

    X=np.arange(data_shape[0])
    Y=np.arange(data_shape[1])
    XX,YY = np.meshgrid(X,Y)
    distance = (XX-x0)**2 + (YY-y0)**2

    rad_mask = (distance > radius_inner**2) & (distance < radius_outer**2)

    Z = data[rad_mask]

    # Mask excess emission in actual data
    data_median = np.median(Z)
    data_std = np.std(Z)

    data_mask = data-data_median < 2 * data_std

    all_mask = rad_mask & data_mask
    if grow>0:
        xval,yval=np.where(~all_mask)
        for x1,y1 in zip(xval, yval):
            all_mask[x1-int(grow/2):x1+int(grow/2),
                y1-int(grow/2):y1+int(grow/2)]=False

    XX = XX[all_mask]-x0
    YY = YY[all_mask]-y0
    Z = data[all_mask]

    X = XX.ravel() ; Y = YY.ravel() ; Z = Z.ravel()

    if test:
        testhdu = fits.PrimaryHDU()
        testdata = data
        testdata[~all_mask]=np.nan
        testhdu.data = testdata
        testhdu.writeto('test.fits', overwrite=True)

    xdata = np.vstack((Y, X))

    if order==1:
        def background(M, a, b, c, d):
            x,y = M
            z = a + b*x + c*x*y + d*y
            return(z)
        params = (np.median(data[~np.isnan(data)]), 1.0, 1.0, 1.0)
    elif order==2:
        def background(M, a, b, c, d, e, f, g):
            x,y = M
            z = a + b*x + c*x*y + d*y + e*x**2 + d*x*y**2 + e*x**2*y +\
                f*y**2 + g*x**2*y**2
            return(z)
        params = (np.median(data[~np.isnan(data)]),
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    else:
        raise Exception(f'STOP!  order={order} not allowed!')


    popt, pcov = curve_fit(background, xdata, Z, params)

    X = np.arange(data_shape[0])-x0
    Y = np.arange(data_shape[1])-y0

    XX, YY = np.meshgrid(X,Y)
    xdata = np.vstack((YY.ravel(), XX.ravel()))

    back_model = background(xdata, *popt)
    back_model = np.reshape(back_model, data_shape)

    back_sub = orig_data - back_model

    return(back_model, back_sub)

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
    parser.add_argument('--idx', default=0, type=int,
        help='HDU index to use for photometry.')
    parser.add_argument('--complex-background', default=False,
        action='store_true', help='Fit with a spatially-varying background '+\
        'around ra/dec')
    parser.add_argument('--order', default=1, type=int,
        help='Polynomial order of the complex background model')
    parser.add_argument('--plots', default=False, action='store_true',
        help='Make diagnostic plots for aperture data.')
    parser.add_argument('--verbose', default=False,
        action='store_true', help='Verbose output.')

    return(parser)

if len(sys.argv)<4:
    print('Usage: aperture.py filename ra dec [options]')
    sys.exit()

filename = sys.argv[1]
if not os.path.exists(filename):
    print('ERROR: file {0} does not exist!'.format(filename))
    print('Exiting...')
    sys.exit()

coord = parse_coord(sys.argv[2], sys.argv[3])
if not coord:
    print('ERROR: could not parse '+\
        f'ra={sys.argv[2]}, dec={sys.argv[3]} into a coordinate')
    print('Exiting...')
    sys.exit()

# This is to prevent argparse from choking if dec was not degrees as float
sys.argv[2] = str(coord.ra.degree) ; sys.argv[3] = str(coord.dec.degree)

parser = add_options()
opt = parser.parse_args()

def get_photometry(file, coord, radius=3, significant_figures=4, use_idx=0,
    use_complex_background=False, background_order=1, make_plots=False,
    verbose=False):

    hdu = fits.open(file)

    try:
        w = wcs.WCS(hdu[use_idx].header)
    except:
        print('ERROR: file {0} does not have WCS.'.format(file))
        print('Exiting...')
        sys.exit()

    header = hdu[use_idx].header
    h = hdu[use_idx].header

    mjd=0.0
    if 'DATE-OBS' in h.keys() and 'TIME-OBS' in h.keys():
        t = Time(h['DATE-OBS']+'T'+h['TIME-OBS'])
        mjd=float('%5.8f'%float(t.mjd))
    elif 'MJD-OBS' in h.keys():
        print('MJD:','%5.8f'%float(h['MJD-OBS']))
        mjd=float('%5.8f'%float(h['MJD-OBS']))
    elif 'DATE_OBS' in h.keys():
        t = Time(h['DATE_OBS'])
        mjd=float('%5.8f'%float(t.mjd))

    if 'CD1_1' in header.keys() and 'CD1_2' in header.keys():
        pscale = np.sqrt(float(header['CD1_1'])**2+float(header['CD1_2'])**2)
        pscale = np.abs(pscale * 3600.)
    elif 'CDELT1' in header.keys():
        pscale = np.abs(float(header['CDELT1']) * 3600.0)
    else:
        print('ERROR: could not parse WCS keywords for {0}'.format(file))
        print('Exiting...')
        sys.exit()

    x,y = wcs.utils.skycoord_to_pixel(coord, w, origin=1, mode='wcs')

    if x < 0 or x > header['NAXIS1'] or y < 0 or y > header['NAXIS2']:
        if verbose:
            print(x,y,'not in',file)
            print('Exiting...')
        else:
            print(file,mjd,np.nan,np.nan)
        sys.exit()


    use_data = hdu[use_idx].data

    # Construct apertures for photometry
    if verbose:
        print(f'Radius is {radius} arcsec')
    aperture = SkyCircularAperture(coord, radius * u.arcsec)

    if make_plots:
        data_shape = use_data.shape

        X=np.arange(data_shape[0])
        Y=np.arange(data_shape[1])
        XX,YY = np.meshgrid(X,Y)

        distance = (XX-x)**2 + (YY-y)**2

        rad_mask = distance < (2*radius/pscale)**2

        Z = use_data[rad_mask].ravel()
        distance = np.sqrt(distance[rad_mask].ravel())

        fig, ax = plt.subplots()
        for rad, dat in zip(distance, Z):
            ax.plot(rad, dat, 'o', color='k')

        ax2 = ax.twiny()
        def pscale_function(x):
            return(pscale * x)

        min_val = np.min(Z)
        max_val = np.max(Z)

        data_range = max_val - min_val
        y_range = [min_val-0.05*data_range, max_val+0.05*data_range]
        x_range = [0., 2.0 * radius / pscale]
        ax.set_ylim(y_range)
        ax.set_xlim(x_range)

        x_ticks = ax.get_xticks()
        xtick_labels = ['%2.2f'%float(val) for val in pscale_function(x_ticks)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(xtick_labels)

        ax2.set_xlabel('Distance from Aperture Center in arcsec')
        ax.set_xlabel('Distance from Apeture Center in Pixels')
        ax.set_ylabel('Pixel Value')

        ax.vlines(radius/pscale,*y_range)

        plt.savefig(file.replace('.fits','.radial.png'))

    # If using complex background, first subtract a complex background model
    # from a region around the source
    if use_complex_background:
        back_model, back_sub = complex_background(hdu[use_idx].data, x, y,
            radius_inner=2*radius/pscale, radius_outer=8*radius/pscale)

        half_size = int(8*radius/pscale)
        x_int = int(np.round(x)) ; y_int = int(np.round(y))

        data_extract = hdu[use_idx].data[y_int-half_size:y_int+half_size,
            x_int-half_size:x_int+half_size]
        back_extract = back_model[y_int-half_size:y_int+half_size,
            x_int-half_size:x_int+half_size]
        backsub_extract = back_sub[y_int-half_size:y_int+half_size,
            x_int-half_size:x_int+half_size]

        outdata = np.array([data_extract, back_extract, backsub_extract])
        outheader = copy.copy(hdu[use_idx].header)
        outheader['CRPIX1']=outheader['CRPIX1']-x_int/2
        outheader['CRPIX2']=outheader['CRPIX2']-y_int/2

        newhdu = fits.PrimaryHDU()
        newhdu.data = outdata
        newhdu.header = outheader

        newhdu.writeto(file.replace('.fits','.stamp.fits'), overwrite=True,
            output_verify='silentfix')

        # Put back into data to continue with the remaining photometry methods
        use_data = back_sub
        newhdu = fits.PrimaryHDU()
        newhdu.data = use_data
        newhdu.header = hdu[use_idx].header
        newhdu.writeto(file.replace('.fits','.sub.fits'), overwrite=True,
            output_verify='silentfix')

    # Do photometry on the input data
    phot_table = aperture_photometry(use_data, aperture,
        wcs=WCS(hdu[use_idx].header), method='exact')
    phot = phot_table['aperture_sum'][0]

    pix_aperture = aperture.to_pixel(w)

    # Now get background estimation
    # As default background aperture use an annulus with inner radius
    # and outer radius
    if verbose:
        print(f'r_in={2*radius/pscale}, r_out={4*radius/pscale}')
    background = CircularAnnulus((x,y), r_in = 2 * radius/pscale,
            r_out = 4 * radius/pscale)
    backmask = background.to_mask(method = 'center')
    if isinstance(backmask, list):
        backmask = backmask[0]

    backdata = backmask.multiply(use_data).flatten()
    # These pixels may be masked, so get rid of them
    mask = backmask.data.flatten()!=0.0
    backdata = backdata[mask]

    mask = ~np.isnan(backdata)
    backdata = backdata[mask]

    # Get sigma-clipped median
    for i in np.arange(10):

        median = np.median(backdata)
        mean = np.mean(backdata)
        std = np.std(backdata)

        mask = np.abs(backdata-median) < 3.0 * std
        backdata = backdata[mask]

    # Rescale the background to the area of the circular aperture
    if verbose:
        print(f'Mean background value {mean}')
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
    elif 'MAGZERO' in hdu[use_idx].header.keys():
        zpt=hdu[use_idx].header['MAGZERO']
    elif 'PHOTZP' in hdu[use_idx].header.keys():
        zpt=hdu[use_idx].header['PHOTZP']
    elif 'FPA.ZP' in hdu[use_idx].header.keys():
        zpt=hdu[use_idx].header['FPA.ZP']
        zpt+=2.5*np.log10(hdu[use_idx].header['EXPTIME'])
    elif 'ORIGZPT' in hdu[use_idx].header.keys():
        zpt=hdu[use_idx].header['ORIGZPT']
        zpt+=2.5*np.log10(hdu[use_idx].header['EXPTIME'])
    elif 'PHOTFLAM' in hdu[use_idx].header.keys():
        PHOTFLAM=hdu[use_idx].header['PHOTFLAM']
        PHOTPLAM=hdu[use_idx].header['PHOTPLAM']
        zpt=-2.5*np.log10(PHOTFLAM)-5*np.log10(PHOTPLAM)-2.408
        #zpt+=2.5*np.log10(hdu[use_idx].header['EXPTIME'])
    elif 'BUNIT' in h.keys() and h['BUNIT']=='MJy/sr':
        zpt=-2.5*np.log10(pscale*pscale*23.5045*1.0e-6/3631.0)
    if 'ZPTMUCER' in hdu[use_idx].header.keys():
        zerr=hdu[use_idx].header['ZPTMUCER']

    if verbose:
        print('zeropoint:',zpt)


    mlimit = -2.5*np.log10(3 * np.sqrt(np.pi * pix_aperture.r**2 * backerr**2))
    mlimit = mlimit + zpt

    if verbose:
        print('Counts, phot, background, backerr:',counts,phot,back,backerr)
    
    fmt='%2.{0}f'.format(significant_figures)

    if np.isnan(mag) or np.isnan(merr):
        mag=np.NaN
        magerr=np.NaN
    else:
        mag=mag+zpt ; magerr=np.sqrt(merr**2+zerr**2)
        mag=float(fmt%mag)
        magerr=float(fmt%magerr)

    mlimit=float(fmt%mlimit)

    data = {'mag': mag,
            'magerr':magerr,
            'mlimit':mlimit,
            'mjd':mjd,
            }
    return(data)


data = get_photometry(filename, coord, radius=opt.radius,
    use_idx=opt.idx, use_complex_background=opt.complex_background,
    background_order=opt.order, make_plots=opt.plots)
if opt.verbose:
    print('Got {0}+/-{1} at {2}, {3} for {4}'.format(data['mag'], data['magerr'],
    coord.ra.degree, coord.dec.degree, filename))
mlimit=data['mlimit']
if opt.verbose:
    print(f'Limiting magnitude is {mlimit}')

print(filename,data['mjd'],data['mag'],data['magerr'])
