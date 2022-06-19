from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.background import Background2D
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup, EPSFModel
from photutils.psf import extract_stars, EPSFBuilder, subtract_psf
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.psf import BasicPSFPhotometry

from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.nddata import NDData
from astropy.stats import gaussian_sigma_to_fwhm
from astropy.table import Table
from astropy.visualization import simple_norm

import matplotlib.pyplot as plt

from scipy.ndimage import rotate
import numpy as np
import sys
import glob
import copy

import warnings
warnings.filterwarnings('ignore')

bkgrms = MADStdBackgroundRMS()
mmm_bkg = MMMBackground()
fitter = LevMarLSQFitter()

def rotate_psf_to_img(psf_file, img_file):
    psf_hdu = fits.open(psf_file)
    img_hdu = fits.open(img_file)

    mask = img_hdu[0].data == img_hdu[0].data[0,0]
    img_hdu[0].data[mask]=np.nan

    psf_ang = img_hdu[0].header['PA_V3']

    # Rotate PSF to the angle of the input image
    psf_rot = rotate(psf_hdu[0].data, float(psf_ang)-90.0)

    return(psf_rot)

def get_epsf(img_file, write_out_back=True, write_out_residual=True,
    write_out_epsf_img=True, flux_percentile_limit=95, iterations=3,
    write_out_epsf_file=True):
    img_hdu = fits.open(img_file)

    std = bkgrms(img_hdu[0].data)
    sigma_psf=1.8*2.0
    iraffind = IRAFStarFinder(threshold=3.5*std,
        fwhm=sigma_psf*gaussian_sigma_to_fwhm,minsep_fwhm=0.01,
        roundhi=5.0, roundlo=-5.0,sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)

    # Get initial set of stars in image with iraffind
    print('Finding stars...')
    stars = iraffind(img_hdu[0].data)

    print('Found {0} stars'.format(len(stars)))
    #flux_percentile_limit = (1.0-500./len(stars))*100
    flux_percentile_limit = 90.0
    print('{0} percentile flux is {1}'.format(flux_percentile_limit,
        np.percentile(stars['flux'], flux_percentile_limit)))
    stars = stars['xcentroid','ycentroid','fwhm','sharpness','roundness',
        'npix','pa','flux','sky']

    mask = (stars['flux'] > np.percentile(stars['flux'],
            flux_percentile_limit)) &\
           (stars['sharpness'] < 0.6) &\
           (stars['roundness'] < 0.1)

    bright = stars[mask]

    m='Masked to {0} stars based on flux, sharpness, roundness'
    print(m.format(len(bright)))

    fwhm = np.median(bright['fwhm'])
    std_fwhm = np.std(bright['fwhm'])
    print('Estimated FWHM={0}+/-{1}'.format('%7.5f'%fwhm,'%7.5f'%std_fwhm))

    mask = (bright['fwhm'] > fwhm-3*std_fwhm) &\
        (bright['fwhm'] < fwhm+3*std_fwhm)
    bright = bright[mask]

    print('Masked to {0} stars based on FWHM'.format(len(bright)))

    bright.sort('flux')
    bright.write(img_file.replace('.fits','.bright.dat'),
        format='ascii.no_header', overwrite=True)

    bkg = Background2D(img_hdu[0].data, (21,21), filter_size=(3,3))
    backsub = img_hdu[0].data - bkg.background

    ndbacksub = NDData(data=backsub)

    backhdu = fits.PrimaryHDU(bkg.background)
    backsubhdu = fits.PrimaryHDU(backsub)

    if write_out_back:
        print('Writing out background and background-subtracted file...')
        back_file = img_file.replace('.fits','.back.fits')
        backsub_file = img_file.replace('.fits','.backsub.fits')

        print('Background file:',back_file)
        backhdu.writeto(back_file, overwrite=True)
        print('Background-subtracted file:',backsub_file)
        backsubhdu.writeto(backsub_file, overwrite=True)

    # Instantiate EPSF
    epsf = None
    for i in np.arange(iterations):
        print('PSF iteration #{0}/{1}'.format(i+1,iterations))
        # Construct stars table from bright
        stars_tbl = Table()
        stars_tbl['x'] = bright['xcentroid']
        stars_tbl['y'] = bright['ycentroid']

        stars = extract_stars(ndbacksub, stars_tbl, size=51)
        print('Extracted {0} stars.  Building EPSF...'.format(len(stars)))

        epsf_builder = EPSFBuilder(oversampling=2, maxiters=5,
            progress_bar=True)
        epsf, fitted_stars = epsf_builder(stars)

        if i+1 < iterations:
            print('\n')
            daogroup = DAOGroup(fwhm)
            photometry = BasicPSFPhotometry(group_maker=daogroup,
                                            bkg_estimator=mmm_bkg,
                                            psf_model=epsf,
                                            fitter=fitter,
                                            fitshape=(51,51))

            target = Table()
            target['x_0'] = bright['xcentroid']
            target['y_0'] = bright['ycentroid']
            target['flux_0'] = bright['flux']

            print('Running photometry on bright stars...')
            result_tab = photometry(image=backsub, init_guesses=target)

            bright['xcentroid'] = result_tab['x_fit']
            bright['ycentroid'] = result_tab['y_fit']
            bright['flux'] = result_tab['flux_fit']

    # We should always write out bright stars
    if iterations>0:
        bright_file = img_file.replace('.fits','.stars.dat')
        with open(bright_file, 'w') as f:
            for row in result_tab:
                data = '{x} {y} {flux} \n'
                f.write(data.format(x=row['x_fit'],y=row['y_fit'],
                    flux=row['flux_fit']))

    if write_out_residual:
        subdata = img_hdu[0].data
        for row in result_tab:
            subdata = subtract_psf(subdata, epsf, Table(row))

        residual_file = img_file.replace('.fits','.residual.fits')
        newhdu = fits.PrimaryHDU(subdata)
        newhdu.writeto('residual.fits', overwrite=True)

    if write_out_epsf_img:
        norm = simple_norm(epsf.data, 'log', percent=99.)
        plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.savefig(img_file.replace('.fits','.epsf.png'))
        plt.clf()

    if write_out_epsf_file:
        hdu = fits.PrimaryHDU(epsf.data)
        hdu.header['FWHM']=fwhm
        hdu.writeto(img_file.replace('.fits','.psf.fits'),overwrite=True)

    # Finally return EPSF
    return(epsf, fwhm)

def run_forced_photometry(img_file, epsf, fwhm, x, y):

    img_hdu = fits.open(img_file)
    bkg = Background2D(img_hdu[0].data, (21,21), filter_size=(3,3))
    backsub = img_hdu[0].data - bkg.background
    ndbacksub = NDData(data=backsub)

    psf = copy.copy(epsf)

    stars_tbl = Table([[x],[y]],names=('x','y'))

    stars = extract_stars(ndbacksub, stars_tbl, size=51)

    stars_tbl['flux'] = np.array([stars[0].estimate_flux()])

    targets = Table()
    targets['x_0'] = stars_tbl['x']
    targets['y_0'] = stars_tbl['y']
    targets['flux_0'] = stars_tbl['flux']

    #psf.x_0.fixed = True
    #psf.y_0.fixed = True

    daogroup = DAOGroup(fwhm)
    photometry = BasicPSFPhotometry(group_maker=daogroup,
                                    bkg_estimator=mmm_bkg,
                                    psf_model=psf,
                                    fitter=fitter,
                                    fitshape=(51,51))

    result_tab = photometry(image=backsub, init_guesses=targets)

    return(result_tab)

get_epsf('wfc3.f110w.ref_0001.drz.fits')
