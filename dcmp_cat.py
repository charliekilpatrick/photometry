# Python 2/3 compatibility
from __future__ import print_function
import numpy as np
import glob,warnings,sys,math,os
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii,fits
from astropy.table import Table,Column
from astropy.wcs import WCS
from astroquery.vizier import Vizier
warnings.filterwarnings('ignore')

# sw.dcmp table names
names = ('Xpos              ',# / X position
         'Ypos              ',# / Y position
         'M                 ',# / PSF fitted magnitude
         'dM                ',# / Error in M
         'flux              ',# / flux in counts
         'dflux             ',# / error in flux
         'type              ',# / Type of object, as defined by dophot
         'peakflux          ',# / peakflux of fit
         'sigx              ',# / sigx of fit
         'sigxy             ',# / sigxy of fit
         'sigy              ',# / sigy of fit
         'sky               ',# /
         'chisqr            ',# / chisqr of object
         'class             ',# / Classification of object, as defined by Sextra
         'FWHM1             ',# / FWHM along minor axis
         'FWHM2             ',# / FWHM along major axis
         'FWHM              ',# / FWHM
         'angle             ',# / Position angle
         'extendedness      ',# / Extendedness parameter (dophot)
         'flag              ',# / flags set by cleanim (hexadecimal)
         'mask              ',# / masks within aperture (mask file or bpm/satura
         'Nmask             ',# / # pixels with bad masks within aperture radius
         'RA                ',# / RA of object in deg
         'Dec               '# / Dec of object in deg
)
converters = {'Xpos':     [ascii.convert_numpy(np.float32)],
              'Ypos':     [ascii.convert_numpy(np.float32)],
              'M':        [ascii.convert_numpy(np.float32)],
              'dM':       [ascii.convert_numpy(np.float32)],
              'flux':     [ascii.convert_numpy(np.float32)],
              'dflux':    [ascii.convert_numpy(np.float32)],
              'peakflux': [ascii.convert_numpy(np.float32)],
              'sigx':     [ascii.convert_numpy(np.float32)],
              'sigxy':    [ascii.convert_numpy(np.float32)],
              'sigy':     [ascii.convert_numpy(np.float32)],
              'chisqr':   [ascii.convert_numpy(np.float32)],
              'FWHM1':    [ascii.convert_numpy(np.float32)],
              'FWHM2':        [ascii.convert_numpy(np.float32)],
              'FWHM':         [ascii.convert_numpy(np.float32)],
              'angle':        [ascii.convert_numpy(np.float32)],
              'extendedness': [ascii.convert_numpy(np.float32)],
              'Nmask':        [ascii.convert_numpy(np.float32)],
              'RA':           [ascii.convert_numpy(np.float32)],
              'Dec':          [ascii.convert_numpy(np.float32)]}
names = [name.strip() for name in names]

def get_unique_objdate (directory = '.'):

    dcmps = glob.glob(directory+'/*.sw.dcmp')

    # Get all object names and observation dates from dcmp
    objs = [fits.getval(dcmp,'OBJECT') for dcmp in dcmps]
    mjds = [fits.getval(dcmp,'MJD-OBS') for dcmp in dcmps]
    exps = [fits.getval(dcmp,'EXPTIME') for dcmp in dcmps]
    sats = [fits.getval(dcmp,'SATURATE') for dcmp in dcmps]
    fils = [fits.getval(dcmp,'FILTER') for dcmp in dcmps]

    # Make a table with dcmp, obj, mjd
    table = Table([dcmps, objs, mjds, exps, sats, fils],
        names=('file', 'obj', 'mjd', 'exptime', 'saturation', 'filter'))

    # Mask out rows with exptime < 45 seconds
    table = table[table['exptime'] > 45]

    # Get unique obj
    unique_obj = list(set(objs))

    # For unique obj, get groups of mjds that are within 0.5 days
    groups = []
    for u_obj in unique_obj:
        rows = table[table['obj']==u_obj]
        for row in rows:
            added = False
            for group in groups:
                if (group['obj'][0]==row['obj'] and
                    abs(group['mjd'][0] - row['mjd']) < 0.5):
                    group.add_row(row)
                    added = True
            if not added:
                groups.append(Table(row))

    return(groups)

def getPS1cat4table (table):
    # Assume a table of observations, check RA/DEC to make sure overlapping
    ra = fits.getval(table['file'][0], 'RA')
    dec = fits.getval(table['file'][0], 'DEC')
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.degree), frame='icrs')
    for file in table['file']:
        ra = fits.getval(file, 'RA')
        dec = fits.getval(file, 'DEC')
        coord = SkyCoord(ra, dec, unit=(u.hourangle, u.degree), frame='icrs')
        if coord.separation(c).degree > 0.4:
            error = 'ERROR: one of the obs for {obj} is not overlapping'
            error = error.format(obj=table['obj'][0])
            print(error)
            sys.exit(1)

    # Download a catalog for the first coordinate with a large enough radius to
    # overlap with all catalogs
    radius = 1
    RAboxsize = DECboxsize = 0.6
    Mmax = 21.0

    # get the maximum 1.0/cos(DEC) term: used for RA cut
    minDec = c.dec.degree-0.5*DECboxsize
    if minDec<=-90.0:minDec=-89.9
    maxDec = c.dec.degree+0.5*DECboxsize
    if maxDec>=90.0:maxDec=89.9

    invcosdec = max(1.0/math.cos(c.dec.degree*math.pi/180.0),
                            1.0/math.cos(minDec  *math.pi/180.0),
                            1.0/math.cos(maxDec  *math.pi/180.0))

    ramin = c.ra.degree-0.5*RAboxsize*invcosdec
    ramax = c.ra.degree+0.5*RAboxsize*invcosdec
    decmin = c.dec.degree-0.5*DECboxsize
    decmax = c.dec.degree+0.5*DECboxsize

    vquery = Vizier(columns=['RAJ2000', 'DEJ2000',
                             'gmag', 'e_gmag',
                             'rmag', 'e_rmag',
                             'imag', 'e_imag',
                             'zmag', 'e_zmag',
                             'ymag', 'e_ymag'],
                    column_filters={'gmag':
                                    ('<%f' % Mmax)},
                    row_limit=100000)
    tbdata = vquery.query_region(c, width=('%fd' % radius),
                catalog='II/349/ps1')[0]
    tbdata.rename_column('RAJ2000', 'ra')
    tbdata.rename_column('DEJ2000', 'dec')
    tbdata.rename_column('gmag', 'PS1_g')
    tbdata.rename_column('e_gmag', 'PS1_g_err')
    tbdata.rename_column('rmag', 'PS1_r')
    tbdata.rename_column('e_rmag', 'PS1_r_err')
    tbdata.rename_column('imag', 'PS1_i')
    tbdata.rename_column('e_imag', 'PS1_i_err')
    tbdata.rename_column('zmag', 'PS1_z')
    tbdata.rename_column('e_zmag', 'PS1_z_err')
    tbdata.rename_column('ymag', 'PS1_y')
    tbdata.rename_column('e_ymag', 'PS1_y_err')

    for row in tbdata:
      matches = tbdata[(tbdata['ra'] - row['ra'])**2 +
                (tbdata['dec'] - row['dec'])**2 < radius**2]

    mask = ((tbdata['ra']<ramax) & (tbdata['ra']>ramin) &
            (tbdata['dec']<decmax) & (tbdata['dec']>decmin))
    tbdata = tbdata[mask]

    # Mask table
    for key in tbdata.keys():
        if key not in ['ra','dec']:
            tbdata[key] = [str(dat) for dat in tbdata[key]]

    return(tbdata)

# Cross-match the Swope catalogs to the PS1 table.  The idea here is that PS1
# has already been cross-matched, so the each row of the table provides a source
# that we can look for in the Swope tables.  The strategy is to iterate through
# each Swope table and add new columns for Swope uBVgri magnitudes to each
# source.  The output is the same format as the input PS1 catalog, but with
# new columns for Swope magnitudes.
def crossmatchSwope (tbdata, table, radius=1./3600):
    # table = Swope dcmp table from above, so loop through each table
    for row in table:
        # Grab dcmp photometry from the sw.dcmp file
        dcmp_table = ascii.read(row['file'], data_start = 1, names=names,
            converters=converters)
        dcmp_table.rename_column('RA', 'ra')
        dcmp_table.rename_column('Dec', 'dec')

        message = 'Adding data for {file}.'
        print(message.format(file=row['file']))

        # Cuts to Swope catalog
        sat = row['saturation']
        dcmp_table = dcmp_table[dcmp_table['type']=='0x00000001'] # DoPhot type
        dcmp_table = dcmp_table[dcmp_table['Nmask']==0] # No masked pixels
        dcmp_table = dcmp_table[dcmp_table['chisqr']<1000.] # Good chisqr fit
        dcmp_table = dcmp_table[dcmp_table['peakflux']<sat] # Not saturated

        # Now update RA/Dec columns with header wcs
        w = WCS(row['file'])
        world_coord = w.wcs_pix2world(zip(dcmp_table['Xpos'],
            dcmp_table['Ypos']), 0)
        dcmp_table['ra'] = map(list, zip(*world_coord))[0]
        dcmp_table['dec'] = map(list, zip(*world_coord))[1]

        # Get new column names for table
        filter_name = 'Swope_'+row['filter']
        error_name = filter_name+'_err'

        # The slow part --- first trying brute force method of checking
        # every single row in tbdata and trying to cross match to a single
        # source in dcmp table.  If 0 sources, add a mask, if more than one
        # source initially ignore
        err=[]
        mag=[]
        for entry in tbdata:
            match = dcmp_table[(dcmp_table['ra'] - entry['ra'])**2 +
                (dcmp_table['dec'] - entry['dec'])**2 < radius**2]
            if (len(match)!=1):
                err.append('--')
                mag.append('--')
            else:
                err.append(str(match['dM'][0]))
                mag.append(str(match['M'][0]))

        tbdata.add_column(Column(np.array(mag), name=filter_name))
        tbdata.add_column(Column(np.array(err), name=error_name))

    return(tbdata)

groups = get_unique_objdate()
tbdata = getPS1cat4table(groups[0])
tbdata = crossmatchSwope(tbdata, groups[0])
name_parts = str(os.path.basename(groups[0]['file'][0])).split('.')
new_name = name_parts[0]+'_'+name_parts[2]+'.cat'
tbdata.write(new_name, format='ascii.fixed_width', overwrite=True)

