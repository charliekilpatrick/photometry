from __future__ import print_function
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np, os, sys, warnings, glob

binned=False

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
                        phottabs[str(photdata[0][0].strip())]=photdata
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

def get_all_tables(table_dir):
    files = None
    candidates = None
    photometry = []
    for file in glob.glob(table_dir+'/*.clusters'):
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


    print(candidates.keys())
    cand_coords = SkyCoord(candidates['RAaverage'],
        candidates['DECaverage'], unit=(u.hour, u.deg))
    sep = coord.separation(cand_coords)
    mask=sep<2.0*u.arcsec

    if len(candidates[mask])==0:
        return(None)
    else:
        idx = np.argmin(sep)
        return(candidates[idx]['ID'])

cand_ra='13:13:01.58'
cand_dec='-19:30:45.11'
cand_coord = SkyCoord(cand_ra, cand_dec, unit=(u.hour, u.deg))

for subdir in ['2021fxy']:
    files, candidates, photometry = get_all_tables(subdir)

    cand_id = get_cand_id(cand_coord, candidates)
    if not cand_id: continue

    outdata = Table([[0.],[''],[0.],[0.],['X'*20]],
        names=('MJD','FILTER','MAG','MAGERR','SOURCE')).copy()[:0]

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
        print(phot)
        if '94' in str(phot): source='LULIN'
        if '4a' in str(phot): source='LCOGT'

        outdata.add_row([float(mjd),filt,float(row['M'])+float(zpt),
            float(row['dM'])+0.01,source])

    if binned:
        minMJD=np.min(outdata['MJD'])-0.01
        maxMJD=np.max(outdata['MJD'])
        days=int(maxMJD-minMJD+1)

        for d in np.linspace(np.min(outdata['MJD']),np.max(outdata['MJD']), days*4):
            for filt in ['u','B','V','g','r','i','z']:
                mask = (outdata['MJD']>d) & (outdata['MJD']<d+0.25) &\
                    (outdata['FILTER']==filt) & (outdata['MAG']<30.0)

                masked_data = outdata[mask]
                if len(masked_data)==0: continue
                elif len(masked_data)==1:
                    print('%5.5f'%masked_data[0]['MJD'],masked_data[0]['FILTER'],
                        '%5.3f'%masked_data[0]['MAG'],'%5.3f'%masked_data[0]['MAGERR'])
                else:
                    flux = 10**(-0.4*(masked_data['MAG'].data-27.5))
                    fluxerr = flux * masked_data['MAGERR'].data/1.086

                    mean_flux = np.mean(flux)
                    mean_fluxerr = np.sqrt(np.sum(1./len(masked_data)*fluxerr**2))

                    mean_mag = -2.5*np.log10(mean_flux)+27.5
                    mean_magerr = mean_fluxerr/mean_flux * 1.086

                    mean_mjd = np.mean(masked_data['MJD'].data)

                    source=masked_data['SOURCE'][0]

                    print('%5.5f'%mean_mjd, masked_data[0]['FILTER'],
                        '%5.3f'%mean_mag, '%5.3f'%mean_magerr)

outdata.sort('MJD')
for row in outdata:
    out='{0} {1} {2} {3}'
    out=out.format('%5.5f'%row['MJD'],row['FILTER'],'%5.3f'%row['MAG'],
        '%5.3f'%row['MAGERR'])
    print(out)
