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
from astropy.stats import sigma_clipped_stats
from scipy.optimize import curve_fit
from scipy.stats import sigmaclip

# photutils dependencies
from photutils import SkyCircularAperture
from photutils import SkyEllipticalAperture
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils import aperture_photometry
from photutils.background import MADStdBackgroundRMS
from photutils.detection import IRAFStarFinder

import matplotlib.pyplot as plt

def curve_of_growth_data(img_file, fwhm_init=5.0, threshold=50.0, 
    diagnostic_plots=True, verbose=True):
    '''
    Function to calculate the curve-of-growth correction for performing aperture photometry
    on objects smaller than the PSF of the image. Uses code from POTPyRI psf.py.

    Inputs:
            img_file [fits]: File to perform photometry on
            fwhm_init [float]: full-width at half-maximum of stars
            threshold [float]: minimum value for source selection
            diagnostic_plots [boolean]: When True, returns plot of curve-of-growth
    Outputs:
            COG_corr []: Curve-of-growth aperture correction factor (in AB magnitudes)

    '''
    # step 1: identify stars
    img_hdu = fits.open(img_file)
    data = img_hdu[0].data

    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(data)
    iraffind = IRAFStarFinder(threshold=threshold*std**2, fwhm=fwhm_init)

    stars = iraffind(data)
    
    mean, median, std_sky = sigma_clipped_stats(img_hdu[0].data, sigma=5.0)
    img_data = img_hdu[0].data - median

    ##############################################################

    # step 2: ensure stars are point sources
    mask = ((stars['xcentroid'] > 50) & (stars['xcentroid'] < (data.shape[1] -51))
                & (stars['ycentroid'] > 50) & (stars['ycentroid'] < (data.shape[0] -51)))
    stars = stars[mask]

    stars = stars['xcentroid', 'ycentroid', 'fwhm', 'sharpness', 'roundness', 
        'npix', 'pa', 'flux', 'sky']

    # mask based on sharpness and roundness
    mask = ((stars['sharpness'] < np.median(stars['sharpness'])+np.std(stars['sharpness'])) &\
                (stars['roundness'] < np.median(stars['roundness'])+3*np.std(stars['roundness'])) &\
                (stars['roundness'] > np.median(stars['roundness'])-3*np.std(stars['roundness'])))
    fwhm_stars = stars[mask]

    # mask based on FWHM
    fwhm_clipped, _, _ = sigmaclip(fwhm_stars['fwhm'])
    fwhm = np.median(fwhm_clipped)
    std_fwhm = np.std(fwhm_clipped)
    mask = (fwhm_stars['fwhm'] > fwhm-3*std_fwhm) &\
        (fwhm_stars['fwhm'] < fwhm+3*std_fwhm)
    fwhm_stars = fwhm_stars[mask]
    fwhm = np.median(fwhm_stars['fwhm'])

    ##############################################################
    
    # step 3: apply circular apertures to stars for a range of FWHM and do photometry
    step_size = 0.1
    radii = np.arange(0.10*fwhm, 8.00*fwhm, step_size)
    
    coords = [(fwhm_stars['xcentroid'][i],fwhm_stars['ycentroid'][i]) for i in range(len(fwhm_stars))]
    
    apertures=[]
    for r in radii:
        apertures.append(CircularAperture(coords, r=r))  # Annuli apertures

    phot_table = aperture_photometry(img_data, apertures)
    phot_table.remove_columns(['id', 'xcenter', 'ycenter']) # new table with only aperture sums
    ##############################################################

    fractions = []
    for row in phot_table:
        flux = [row[k] for k in phot_table.keys() if 'aperture_sum' in k]
        flux = np.array(flux)
        fraction = flux / np.max(flux)
        fractions.append(fraction)

    # step 4: make curve_of_growth plot to check it worked
    if diagnostic_plots:
        fig,ax = plt.subplots()
        mean_fraction = []
        fraction_err = []
        for i in np.arange(len(radii)):
            mean_fraction.append(np.mean([f[i] for f in fractions]))
            fraction_err.append(np.std([f[i] for f in fractions]))

        ax.errorbar(radii, mean_fraction, yerr=fraction_err) 
        # phot measurement vs aperture size

        plt.xlabel('Aperture radius [pix]')
        plt.ylabel('Encircled energy fraction')
        plt.show()

    ##############################################################

    # step 5: calculate curve_of_growth correction factor
    # COG_corr = flux @ r<FWHM / flux @ FWHM

    # step 6: convert to a magnitude
    # flux to mag conversion here?
    # COG_corr = X magnitudes

    return(radii, mean_fraction, fraction_err)

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
        help='Input filename')
    parser.add_argument('ra', type=str,
        help='Right ascension to place aperture')
    parser.add_argument('dec', type=str,
        help='Declination to place aperture')

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
    parser.add_argument('--elliptical', nargs=3, type=float, default=None,
        help='semi-major axis (arcsec), semi-minor  (arcsec), and'+\
        ' position angle (in deg) from positive x axis.')
    parser.add_argument('--curve_of_growth', default=False,
        action='store_true', help='Perform curve-of-growth correction.')
    parser.add_argument('--fwhm', default=3.0, type=float,
        help='FWHM of the PSF in pixels.')

    return(parser)

def get_photometry(file, coord, radius=3, significant_figures=4, use_idx=0,
    use_complex_background=False, background_order=1, make_plots=False,
    verbose=False, elliptical=None, curve_of_growth=False, fwhm=3):

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
        mjd=float('%5.8f'%float(h['MJD-OBS']))
    elif 'DATE_OBS' in h.keys():
        t = Time(h['DATE_OBS'])
        mjd=float('%5.8f'%float(t.mjd))

    if verbose: print('MJD:','%5.8f'%float(h['MJD-OBS']))

    pscale = np.mean(wcs.utils.proj_plane_pixel_scales(w)) * 3600.0
    if verbose: print(f'pixel scale={pscale}')

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

    if elliptical:
        semimajor = elliptical[0]
        semiminor = elliptical[1]
        position_angle = elliptical[2]

        aperture = SkyEllipticalAperture(coord, semimajor * u.arcsec,
            semiminor * u.arcsec, position_angle * u.deg)
    else:
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

    # do curve of growth correction
    if curve_of_growth:
        radii, fraction, fraction_err = curve_of_growth_data(file, fwhm_init=fwhm, verbose=verbose)

    # Do photometry on the input data
    phot_table = aperture_photometry(use_data, aperture,
        wcs=WCS(hdu[use_idx].header), method='exact')
    phot = phot_table['aperture_sum'][0]

    pix_aperture = aperture.to_pixel(w)

    # Now get background estimation
    # As default background aperture use an annulus with inner radius
    # and outer radius
    if verbose: print(f'r_in={2*radius/pscale}, r_out={4*radius/pscale}')

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
        zpt+=2.5*np.log10(hdu[use_idx].header['EXPTIME'])
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
            'mjd':mjd,}
    return(data)


if __name__=='__main__':
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

    data = get_photometry(filename, coord, radius=opt.radius,
        use_idx=opt.idx, use_complex_background=opt.complex_background,
        background_order=opt.order, make_plots=opt.plots, elliptical=opt.elliptical,
        verbose=opt.verbose, curve_of_growth=opt.curve_of_growth)
    if opt.verbose:
        print('Got {0}+/-{1} at {2}, {3} for {4}'.format(data['mag'],data['magerr'],
            coord.ra.degree,coord.dec.degree,filename))
    mlimit=data['mlimit']
    if opt.verbose:
        print(f'Limiting magnitude is {mlimit}')
    else:
        print('{0} {1}'.format(data['mag'], data['magerr']))
