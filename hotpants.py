import glob
import os
import copy

opt1={'ko':0.01,'bgo':0.01,'c':'t','n':'t'}

def run_hotpants(files, temp, kwargs={}, clobber=False):
    infiles = copy.copy(files)

    outfiles = []
    for i,file in enumerate(files):
        if file==temp: continue

        cmd = 'hotpants -inim {0} -tmplim {1} -outim {2} {3}'

        opt1['il']=-1000
        opt1['iu']=50000
        opt1['tl']=-1000
        opt1['tu']=50000
        for key in kwargs.keys():
            opt1[key]=kwargs[key]

        opt=''
        for key in opt1.keys():
            opt += '-'+key.strip()+' '+str(opt1[key])+' '

        filt = file.split('.')[1]
        obj = file.split('.')[0]

        outfile1='{0}_{1}.diff.fits'
        outfile1=outfile1.format(file.replace('.fits',''),
            temp.replace('.fits',''))

        cmd1 = cmd.format(file, temp, outfile1, opt)
        print(cmd1)
        if not os.path.exists(outfile1) or clobber:
            os.system(cmd1)

        outfiles.append(outfile1)

    return(infiles, outfiles)


