#!/usr/bin/env python3
# Python 2/3 compatibility
import requests, sys, os
from astropy.io import ascii
from astropy.time import Time
from astropy.table import Table, Column, unique
from dateutil.parser import parse
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

import pysynphot as S
S.setref(area = 25.0 * 10000)

def add_options(parser=None, usage=None):
    import argparse
    if parser == None:
        parser = argparse.ArgumentParser(usage=usage,conflict_handler='resolve')

    parser.add_argument('filename', type=str,
        help='Right ascension to reduce the HST images')
    parser.add_argument('--filter','-f', type=str, default=None,
        help='Comma-separated list of filters to plot')
    parser.add_argument('--t_range', type=str, default=None,
        help='Comma-separated list of time range to plot (e.g., 58500,58600)')
    parser.add_argument('--update', default=False, action='store_true',
        help='Instead of plotting once, refresh plot on user input')
    parser.add_argument('--compare', default='', type=str,
        help='Plot data from input file with different symbol for comparison.')
    parser.add_argument('--mask', default=False, action='store_true',
        help='Mask out nan and negative values instead of throwing an error.')

    return(parser)

# For the purposes of ordering, roughly the wavelength of
# each filter contained in YSE PZ.
filter_wavelengths = {
    'u': 3543,'g': 4770,'r': 6231,'i': 7625,'z': 8660,'y': 9620,'U': 3640,
    'B': 4420,'V': 5400,'R': 6349,'I': 8797,'up': 3540,'gp': 4750,'rp': 6222,
    'ip': 7632,'zp': 9049,'Y': 10200,'J': 12200,'H': 16300,'K': 21900,
    'UVW1': 2600,'UVM2': 2246,'UVW2': 1928, 'zs': 8260,
}

temp_wave = 1.0 + np.arange(60000)

telescopes = {
    'swope': ['u','B','V','g','r','i'],
    'lcogt': ['U','B','V','R','I','up','gp','rp','ip','zs'],
    'hst': [],
    'spitzer': ['I1','I2'],
    'swift': ['U','B','V','UVM2','UVW1','UVW2','W'],
    'atlas': ['orange','cyan'],
    'ztf': [],
}

filter_dir = 'FILTERS/'
inst_dict = {'uvot':'Swift/UVOT',
             'sinistro': 'LCOGT 1m/Sinistro',
             'ztf cam': 'ZTF'}

def is_number(num):
    try:
        num = float(num)
    except ValueError:
        return(False)
    return(True)

def get_filter_file(telescope, filt, instrument=None, wavelength=None):
    telescope = telescope.upper()
    tel_key = ''

    if not wavelegth:
        wavelength = temp_wave

    check = [telescope in key.upper() for key in telescopes.keys()]
    if any(check):
        if len(np.array(list(telescope.keys()))[check])>1:
            error = 'ERROR: ambiguous telescope name {0}\n'
            error += 'Provide a telescope name from {1}'
            print(error.format(telescope, list(telescope.keys())))
            return(None)
        else:
            tel_key = np.array(list(telescope.keys()))[check][0]
            possible_filts = telescopes[key]

    check = [key.upper() in telescope for key in telescope.keys()]
    if any(check):
        if len(np.array(list(telescope.keys()))[check])>1:
            error = 'ERROR: ambiguous telescope name {0}\n'
            error += 'Provide a telescope name from {1}'
            print(error.format(telescope, list(telescope.keys())))
            return(None)
        else:
            tel_key = np.array(list(telescope.keys()))[check][0]
            possible_filts = telescopes[key]

    # Now try to parse filter
    if filt in possible_filts:
        # Construct filter file name from telescope.filter
        filename = '{0}.{1}.dat'.format(tel_key.upper(), filt)

        fullfilename = filter_dir + filename

        # Open file and parse into passband
        wave, transmission = np.loadtxt(fullfilename, unpack=True, dtype=float)

        # Load into pysynphot arraybandpass object
        bpmodel = S.ArrayBandpass(wave, transmission,
                name=filename, waveunits='Angstrom')

        return(bpmodel)

def plot_table(table, trim_outliers=True, plot_time='time', plot_mag='mag',
    absolute=False, plot_bands=[], t_range=[], compare=''):
    uniq_filters = np.unique(table['band'])
    uniq_filters = sorted(uniq_filters, key=lambda f: filter_wavelengths[f])

    if plot_bands:
        uniq_filters = np.array([u for u in uniq_filters if u in plot_bands])

    mask = np.array([t['band'] in uniq_filters for t in table])
    table = table[mask]

    # Get an array of colors for plotting different bands
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    # Get offsets if needed
    x = np.linspace(0.0, 1.0, len(uniq_filters))

    if trim_outliers:
        meanx = np.mean([Time(t).mjd for t in table[plot_time]])
        stdx = np.std([Time(t).mjd for t in table[plot_time]])
        match = [np.abs(Time(t).mjd - meanx)<10*stdx for t in table[plot_time]]
        table = table[match]

        # Trim outliers for absolute/apparent mag
        if absolute: table = table[table[plot_mag] < 0.]
        else: table = table[table[plot_mag] > 0.]

        # Trim large errors
        table = table[table[plot_mag+'_err']<0.2]

    for filt in uniq_filters:
        subtable = table[table['band']==filt]

        if is_number(subtable[plot_time][0]):
            form='mjd'
        else:
            form=None

        time = [Time(t, format=form).mjd for t in subtable[plot_time]]
        mag = [m for m in subtable[plot_mag]]
        err = [e for e in subtable[plot_mag+'_err']]

        plt.plot(time, mag, 'o', linestyle='None', label=filt)
        plt.errorbar(time, mag, yerr=err, linestyle='None')

    if compare:
        ctable = get_data(compare)

        for filt in uniq_filters:
            subtable = ctable[ctable['band']==filt]
            if len(subtable)==0:
                continue

            if is_number(subtable[plot_time][0]):
                form='mjd'
            else:
                form=None

            time = [Time(t, format=form).mjd for t in subtable[plot_time]]
            mag = [m for m in subtable[plot_mag]]
            err = [e for e in subtable[plot_mag+'_err']]

            plt.plot(time, mag, '*', linestyle='None', label=filt)
            plt.errorbar(time, mag, yerr=err, linestyle='None')

    # y limit adjustment
    ymax = np.max(table[plot_mag]+table[plot_mag+'_err'])
    ymin = np.min(table[plot_mag]-table[plot_mag+'_err'])
    yran = (ymax - ymin)

    if len(t_range)==2:
        plt.xlim(t_range[0], t_range[1])

    plt.ylim(ymax+0.05*yran, ymin-0.05*yran)
    plt.legend(loc='upper right')

def sanitize_photometry_table(table, write=True):

    # Get rid of bands with Unknown data
    match = ['Unknown' not in band for band in table['band']]
    table = table[match]

    # Get rid of rows with None
    match = [mag is not None for mag in table['mag']]
    table = table[match]

    if 'instrument' not in table.keys():
        col = Column([None]*len(table), name='instrument')

        for i,row in enumerate(table):
            band_data = row['band'].split('-')
            if 'ZTF' in band_data:
                filt = band_data[2].strip()
            else:
                filt = band_data[-1].strip()

            if band_data[0].strip().lower() in inst_dict.keys():
                inst = inst_dict[band_data[0].strip().lower()]
            else:
                inst = band_data[0].strip()

            col[i] = inst
            table[i]['band'] = filt

        # Add column
        table.add_column(col)

    # Get rid of unknown filters
    match = [f in filter_wavelengths.keys() for f in table['band']]
    table = table[match]

    if write:
        if 'name' in table.meta.keys():
            name = table.meta['name']
            ascii.write(table, name+'.phot')

    return(table)

def yse_pz_phot(objname, check=True):

    if os.path.exists(objname+'.phot'):
        table = ascii.read(objname+'.phot')
        return(table)

    url = 'https://ziggy.ucolick.org/yse/download_data/'

    data = requests.get(url + objname)

    if 'Server Error' in data.text:
        error = 'ERROR: could not get data for {obj} from YSE PZ\n'
        error += 'Exiting...'
        print(error.format(obj=objname))
        sys.exit()

    data = data.json()

    table_data = []

    for group in data[objname]['photometry']:
        for p in group['data']:
            datapoint = p['fields']
            mag = datapoint['mag']
            err = datapoint['mag_err']
            data_time = Time(datapoint['obs_date'])
            band = datapoint['band']

            table_data.append([data_time, band, mag, err])

    table_data = list(map(list, zip(*table_data)))
    photometry_table = Table(table_data, names=('time','band','mag','mag_err'))
    photometry_table.meta['name'] = objname

    return(photometry_table)

# Assume relative_date is in the same units as table['time']
# and calculate the relative phase for each row, then add a 'phase' column
def add_phase_from_date(table, relative_date, z=None):

    # Don't need to re-add phase key if it's already there
    if 'phase' in table.keys():
        return(table)

    # Try to parse the relative_date variable
    try:
        rel_date = parse(relative_date)
    except dateutil.parser._parser.ParserError:
        warning = 'WARNING: could not parse date: {0}.  Enter formatted date.'
        print(warning.format(relative_date))
        return(table)

    scale=1.
    if z:
        scale = 1.+z

    phase=[]
    for row in table:
        phase.append((row['time']-rel_date)/scale)

    col = Column(phase, name='phase')
    table.add_column(col)

    return(table)

# Returns a list of unique telescope/filter pairs for the given phottable
def get_unique_telescope_filter(phottable):
    filt_unique = unique(phottable, keys=['band','instrument'])
    filt_unique = filt_unique['band', 'instrument']

    return(filt_unique)

# Add filter transmission function to a list of unique filter/instrument pairs.
# The transmission function will be interpolated over a specific wavelength
# range and padded with zeros so all transmission functions cover the same
# wavelength range and have the same wavelength resolution
def add_transmission(bands):

    passbands = []
    for row in bands:
        filt = row['band']
        inst = row['instrument']

        bpmodel = get_filter_file(inst, filt)

        passbands.append(bpmodel)

    col = Column(passbands, name='passband')
    bands.add_column(col)

    return(bands)

# Corrects the photometry in the table for extinction with provided E(B-V) and
# optional reddening law.  If no reddening law is provided, we assume the values
# in the function, which are taken from Schlafly & Finkbeiner 2011 for R_V=3.1.
def correct_for_reddening(phottable, ebv, reddening=None):
    if 'MW_reddening' in phottable.meta.keys():
        if phottable.meta['MW_reddening']:
            return(phottable)

    if not reddening:
        reddening={'u':4.239,'g':3.303,'r':2.285,'i':1.698,'z':1.263,
           'up':4.239,'gp':3.303,'rp':2.285,'ip':1.698,'zp':1.263,
           'zs': 1.3,
           'cyan':2.742,'orange':1.921,'G':2.171,
          'U':4.107,'B':3.641,'V':2.682,'R':2.119,'I':1.516,
          'J':0.709,'H':0.449,'K':0.302}

    # Check if MW reddening flag is in table and if it is "True"
    if 'MW_reddening' in phottable.meta.keys():
        if phottable.meta['MW_reddening']:
            print('WARNING: input table already corrected for MW reddening')
            return(phottable)

    # Check that all bands are in the reddening law dictionary
    if any([band not in reddening.keys() for band in phottable['band']]):
        # Get the list of bands that are not in reddening.keys()
        bands=np.unique(phottable['band'])
        print('ERROR: some bands are not in the reddening law table:')
        for b in bands:
            if b not in reddening.keys():
                print(b)
        print('Update reddening table and retry')
        return(phottable)

    for i,row in enumerate(phottable):
        mag = row['mag']
        mag = mag - ebv * reddening[row['band']]
        phottable[i]['mag']=mag

    phottable.meta['MW_reddening']=True
    return(phottable)

def get_data(filename, mask=False):
    table = None

    if not mask:
        # Check for nan and negative magnitude values before plotting
        with open(filename, 'r') as f:
            for line in f:
                mag = line.split()[2]
                if np.isnan(float(mag)):
                    raise Exception(f'ERROR: {filename} contains nan values')
                elif float(mag)<0.0:
                    raise Exception(f'ERROR: {filename} contains negative '+\
                        '(uncalibrated) magnitude values')

    if os.path.exists(filename):
        table = ascii.read(filename, names=('time','band','mag','mag_err'))
        mask_col = ~np.isnan(table['mag']) & (table['mag']>0.0)
        table = table[mask_col]
    else:
        table = yse_pz_phot(filename, check=True)
        table = sanitize_photometry_table(table, write=True)

    if table is None:
        raise Exception('ERROR: no input data from file or YSE-PZ!')

    return(table)


# Parse arguments
if len(sys.argv)<2:
    usage = 'Usage: lightcurve.py obj/file'
    print(usage)
    sys.exit()

parser = add_options()
args = parser.parse_args()

plot_bands=[]
if args.filter is not None:
    plot_bands = args.filter.split(',')

t_range = []
if args.t_range is not None:
    t_range = [float(s) for s in args.t_range.split(',')]

if args.update:
    stop = False
    while not stop:
        table = get_data(args.filename, mask=args.mask)
        plt.clf()
        plot_table(table, trim_outliers=False, plot_bands=plot_bands,
            t_range=t_range, compare=args.compare)
        plt.show()

        x = input('Remake the plot ([y]/n)? ')
        if 'n' in x:
            stop = True

else:
    table = get_data(args.filename, mask=args.mask)
    plot_table(table, trim_outliers=False, plot_bands=plot_bands,
            t_range=t_range, compare=args.compare)
    plt.show()
