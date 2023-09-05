import logging
import os
import pickle
from datetime import datetime
from glob import glob
from shutil import copy2, move

import numpy as np
from netCDF4 import Dataset
from scipy.linalg import svd

import matplotlib.pyplot as plt

logger = logging.getLogger("irs_logger")


def pca_expand(coef, PC):

    (nX, ntr) = coef.shape
    nvals = len(PC['m'])

    x = np.dot(coef, PC['u'][:, 0:ntr].T) + np.full((nX, 1), 1.) @ np.array([PC['m']])

    return x


def pca_ind(evals, nsamp, doplot=False):
    """
    Abstract:
        This script takes as input an array of eigenvalues and computes from them
        the IND, IE, RE, PCV, and other parameters.  But its main function is to compute
        the number of eigenvectors that should be used in the reconstruction of the PCA
        data, based upon the Error Indicator Function (IND).

    Reference:
        Turner, D. D., R. O. Knuteson, H. E. Revercomb, C. Lo, and R. G. Dedecker, 2006:
        Noise Reduction of Atmospheric Emitted Radiance Interferometer (AERI) Observations Using
        Principal Component Analysis. J. Atmos. Oceanic Technol., 23, 1223–1238,
        https://doi.org/10.1175/JTECH1906.1.

    Author:
        Dave Turner
        SSEC / University of Wisconsin - Madison

        Ported to Python by Tyler Bell (CIWRO/NSSL) - Sept 2020


    :param evals: Array of eigenvalues
    :param nsamp: Number of samples
    :return:
    :rtype:
    """

    c = len(evals)

    f_ie = np.zeros(c-1)
    f_re = np.zeros(c-1)
    f_xe = np.zeros(c-1)
    f_ind = np.zeros(c-1)
    f_pcv = np.zeros(c-1)
    for n in range(c-1):
        f_ie[n] = np.sqrt( (n*np.sum(evals[n+1:c])) / (nsamp*c*(c-n)) )
        f_re[n] = np.sqrt( np.sum(evals[n+1:c]) / (nsamp*(c-n)) )
        f_xe[n] = np.sqrt( np.sum(evals[n+1:c]) / (nsamp * c) )
        f_pcv[n] = 100. * np.sum(evals[0:n+1]) / np.sum(evals)
        f_ind[n] = f_re[n] / (c-n)**2.

    idx = np.arange(0, c-1)

    if doplot:
        import matplotlib.pyplot as plt
        bar = np.where(idx >= 5)
        foo = np.argmin(f_ie[bar])
        print(f"The IE optimal number of PCs to use is {idx[bar][foo]}")
        foo = np.argmin(f_ind)
        print(f"The IND optimal number of PCs to use is {idx[foo]}")

        plt.subplot(2, 2, 1)
        plt.semilogx(idx, f_re)
        plt.title("Real Error")
        plt.subplot(2, 2, 2)
        plt.semilogx(idx, f_ie)
        plt.title("Imbedded Error")
        plt.subplot(2, 2, 3)
        plt.loglog(idx, f_ind)
        plt.title("Error Indicator Function")
        plt.show()


    foo = np.argmin(f_ind)
    xe = f_xe[foo]
    re = f_re[foo]
    pcv = f_pcv[foo]

    return idx[foo]


def pca_project(x, ntr, pc):
    """
    Abstract:
        Routine to project (compress) along eigenvectors determined from the
        principal component analysis.

    Original Author:
        Paolo Antonelli and Raymond Garcia
            University of Wisconsin - Madison

    Ported into IDL by:
        Dave Turner
            University of Wisconsin - Madison
            Pacific Northwest National Laboratory

    IDL version ported into Python by:
        Tyler Bell
            OU CIMMS/NSSL

    :param x: A two-dimensional matrix, with time as the second dimension
    :param ntr: Numberprincipal components to use in the projection
    :param pc: Dictionary containing PCA reconstruction matrix
    :return: Columns of principal component coefficients
    """

    (nX, nvals) = x.shape

    coef = x - np.full((1, nX), 1.).T @ np.array([pc['m']])
    coef = np.dot(pc['u'][:, 0:ntr].T, coef.T).T

    return coef


def create_irs_noise_filter(files, sdir, sfields, pcs_filename=None, 
                            use_median_noise=False, sky_view_angle=None):
    """
    Abstract:
        This function is designed to create the PCA the noise filter from IRS obvservations.  
        It uses a Principal Component Analysis technique published in Turner et al.
        JTECH 2006.  It is designed to use either ARM-formatted AERI, dmv-ncdf
        formatted AERI, or ASSIST data. 

    Author: Tyler Bell (CIWRO/NSSL, 2023) 2023
        Based on code from Dave Turner
        
    Reference:
        Turner, D. D., R. O. Knuteson, H. E. Revercomb, C. Lo, and R. G. Dedecker, 2006:
        Noise Reduction of Atmospheric Emitted Radiance Interferometer (AERI) Observations Using
        Principal Component Analysis. J. Atmos. Oceanic Technol., 23, 1223–1238,
        https://doi.org/10.1175/JTECH1906.1.

    """
    
    # Read in the radiance data
    for i, fn in enumerate(sorted(files)):
        logger.debug(f'Loading {fn}')

        nc = Dataset(fn)

        if 'wnum' in nc.variables.keys():
            wnum = nc['wnum'][:]
        elif 'wnum1' in nc.variables.keys():
            wnum = nc['wnum1'][:]
        else:
            logger.error(" Unable to find either wavenumber field in the data - aborting")
            return

        rad = nc['mean_rad'][:]
        foo = np.where(wnum < 3000)
        wnum = wnum[foo]
        rad = rad[:, foo]
        bt = nc['base_time'][0]
        mirror = nc['sceneMirrorAngle'][:]

        try:
            to = nc['time_offset'][:]
        except IndexError:  # ASSISTs are different...
            bt *= 1e-3
            to = nc['time'][:]

        try:
            qcflag = nc['missingDataFlag'][:]
        except IndexError:  # ASSIST doesn't have this... TODO - Add hatch closed filter instead and stuff from Issue #2
            qcflag = np.zeros_like(to)
        
        nc.close()

        if sky_view_angle is not None:
            if((sky_view_angle < 3) | (sky_view_angle > 357)):
                bar = np.where((mirror < 3) | (mirror > 357))[0]
            else:
                bar = np.where((sky_view_angle-3 < mirror) & (mirror < sky_view_angle+3))[0]

            to = to[bar]
            rad = rad[bar]
            qcflag = qcflag[bar]

        if i == 0:
            xsecs = bt+to
            xrad = np.squeeze(rad.T)
            xqcflag = qcflag
        else:
            xsecs = np.append(xsecs, bt+to)
            xrad = np.append(xrad, np.squeeze(rad.T), axis=1)
            xqcflag = np.append(xqcflag, qcflag)

    secs = xsecs
    rad = np.squeeze(np.transpose(xrad))
    qcflag = xqcflag
    # Now read in the summary files
    times = np.array([datetime.utcfromtimestamp(d) for d in xsecs])
    yyyymmdd = [d.strftime("%Y%m%d") for d in times]

    for ymd in np.unique(yyyymmdd):
        sfiles = glob(os.path.join(sdir, f'*sum*{ymd}*.cdf'))
        for j, sfile in enumerate(sorted(sfiles)):
            nc = Dataset(sfile)
            nwnum = nc[sfields[0]][:]
            nrad = nc[sfields[1]][:]
            mirror = nc['sceneMirrorAngle'][:]
            bt = nc['base_time'][0]
            
            try:
                to = nc['time_offset'][:]
            except IndexError:  # ASSISTs are different...
                bt *= 1e-3
                to = nc['time'][:]
            
            nc.close()

            if sky_view_angle is not None:
                if((sky_view_angle < 3) | (sky_view_angle > 357)):
                    bar = np.where((mirror < 3) | (mirror > 357))[0]
                else:
                    bar = np.where((sky_view_angle-3 < mirror) & (mirror < sky_view_angle+3))[0]

                to = to[bar]
                nrad = nrad[bar]

            if j == 0:
                xsecs = bt + to
                xnrad = nrad.T
            else:
                xsecs = np.append(xsecs, bt + to)
                xnrad = np.append(xnrad, nrad.T)

    nsecs = xsecs
    nrad = np.transpose(xnrad)

    # Quick check for consistency
    if len(nsecs) < len(secs):
        ni = len(secs) - len(nsecs)
        logger.debug(f"Adding summary samples: {ni}")
        nrad = np.append(nrad, np.array([nrad[-1, :] for k in range(ni)]), axis=0)

    elif len(nsecs) > len(secs):
        foo = np.where(nsecs <= max(secs)+1)
        if len(foo[0]) != len(secs):
            logger.error("Problem with time matching")
            return
        nsecs = nsecs[foo]
        nrad = nrad[foo, :]

    osecs = secs
    orad = rad.transpose()
    onrad = nrad.transpose()
    oqc = qcflag

    logger.info(f"Number of samples: {len(secs)}; Number of spectral channels: {len(wnum)}")

    # Select only the good data (simple check)
    if min(wnum) < 600:
        logger.debug("I believe I'm working with ch1 data")
        foo = np.where(wnum >= 900)
        minv = -2
        maxv = 105
    elif max(wnum) > 2800:
        logger.debug("I believe I'm working with ch2 data")
        foo = np.where(wnum >= 2550)
        minv = -0.2
        maxv = 2.0
    else:
        logger.error("Unable to determine the channel...")
        return

    good = np.where((minv <= rad[:, foo[0][0]]) & (rad[:, foo[0][0]] < maxv) & (oqc == 0))
    logger.debug(f"There are {len(good[0])} good samples out of {len(secs)}")

    # If the number of good spectra is too small, then we should abort
    if len(good[0]) <= 3*len(wnum):
        logger.critical("There are TOO FEW good spectra for a good PCA noise filter (need > 3x at least")
        logger.critical("Aborting")
        return

    # If the keyword is set, use the median noise spectrum instead of the real noise spectrum
    if use_median_noise:
        logger.info("Using the median noise specturm, not the true sample noise spectrum")
        for i in range(nwnum):
            nrad[:, i] = np.median(nrad[:, i])

    # Normalize the data
    for i in range(len(secs)):
        rad[i, :] = rad[i, :] / np.interp(wnum, nwnum, nrad[i, :])

    # Generate the PCs
    pcwnum = wnum.copy()

    # Compute mean spectrum
    m = np.mean(rad[good], axis=0)

    c = np.cov(rad[good].T)

    logger.debug("Computing the SVD")
    u, d, v = svd(c)
    PC = {'u': u, 'd': d, 'm': m}

    # Determine the number of eigenvectors to use
    nvecs = pca_ind(d, len(good), doplot=False)
    logger.info(f"Number of e-vecs used in reconstruction: {nvecs}")
    pca_comment = f"{len(d)} total PCs, {nvecs} considered significant, derived from {len(good)} " \
                    f"time samples, computed on {datetime.utcnow().isoformat()}"

    gsecs = secs[good]
    data = {'gsecs': gsecs, 'nvecs': nvecs, 'pcwnum': pcwnum, 'pca_comment': pca_comment,
            'PC': PC}
    with open(pcs_filename, 'wb') as fh:
        pickle.dump(data, fh)

    logger.info("DONE creating the PCs and storing them. The NF was not applied yet")

    return


def apply_irs_noise_filter(files, sdir, sfields, tdir, odir, pcs_filename=None, use_median_noise=False):
    """
    Abstract:
        This function is designed to apply the noise filter to IRS observations
        and place the noise-filtered data into the same netCDF file.  It uses
        a Principal Component Analysis technique published in Turner et al.
        JTECH 2006.  It is designed to use either ARM-formatted AERI, dmv-ncdf
        formatted AERI, or ASSIST data.  

    Author: Tyler Bell (CIWRO/NSSL, 2023) 2023
        Based on code from Dave Turner

    """
    
    logger.debug("Copying files to temporary directory...")
    new_files = []
    for fn in files:
        copy2(fn, tdir)
        new_files.append(os.path.join(tdir, os.path.basename(fn)))
    
    # Read in the radiance data
    os.chdir(tdir)
    files = new_files
    for i, fn in enumerate(sorted(files)):
        logger.debug(f'Loading {fn}')

        nc = Dataset(fn)

        if 'wnum' in nc.variables.keys():
            wnum = nc['wnum'][:]
        elif 'wnum1' in nc.variables.keys():
            wnum = nc['wnum1'][:]
        else:
            logger.error(" Unable to find either wavenumber field in the data - aborting")
            logger.error(" Unable to find either wavenumber field in the data - aborting")
            return

        rad = nc['mean_rad'][:]
        foo = np.where(wnum < 3000)
        wnum = wnum[foo]
        rad = rad[:, foo]
        bt = nc['base_time'][0]
        mirror = nc['sceneMirrorAngle'][:]

        try:
            to = nc['time_offset'][:]
        except IndexError:  # ASSISTs are different...
            bt *= 1e-3
            to = nc['time'][:]

        try:
            qcflag = nc['missingDataFlag'][:]
        except IndexError:  # ASSIST doesn't have this... TODO - Add hatch closed filter instead and stuff from Issue #2
            qcflag = np.zeros_like(to)
        
        nc.close()

        # NOTE: I am not removing the blackbody views from ASSIST data when applying the filter
        '''
        if sky_view_angle is not None:
            if((sky_view_angle < 3) | (sky_view_angle > 357)):
                bar = np.where((mirror < 3) | (mirror > 357))[0]
            else:
                bar = np.where((sky_view_angle-3 < mirror) & (mirror < sky_view_angle+3))[0]

            to = to[bar]
            rad = rad[bar]
            qcflag = qcflag[bar]
        '''

        if i == 0:
            xsecs = bt+to
            xrad = np.squeeze(rad.T)
            xqcflag = qcflag
        else:
            xsecs = np.append(xsecs, bt+to)
            xrad = np.append(xrad, np.squeeze(rad.T), axis=1)
            xqcflag = np.append(xqcflag, qcflag)

    secs = xsecs
    rad = np.squeeze(np.transpose(xrad))
    qcflag = xqcflag
    # Now read in the summary files
    times = np.array([datetime.utcfromtimestamp(d) for d in xsecs])
    yyyymmdd = [d.strftime("%Y%m%d") for d in times]

    for ymd in np.unique(yyyymmdd):
        sfiles = glob(os.path.join(sdir, f'*sum*{ymd}*.cdf'))
        for j, sfile in enumerate(sorted(sfiles)):
            nc = Dataset(sfile)
            nwnum = nc[sfields[0]][:]
            nrad = nc[sfields[1]][:]
            mirror = nc['sceneMirrorAngle'][:]
            bt = nc['base_time'][0]
            
            try:
                to = nc['time_offset'][:]
            except IndexError:  # ASSISTs are different...
                bt *= 1e-3
                to = nc['time'][:]
            
            nc.close()
            
            # NOTE: I am not removing the blackbody views from ASSIST data when applying the filter
            '''
            if sky_view_angle is not None:
                if((sky_view_angle < 3) | (sky_view_angle > 357)):
                    bar = np.where((mirror < 3) | (mirror > 357))[0]
                else:
                    bar = np.where((sky_view_angle-3 < mirror) & (mirror < sky_view_angle+3))[0]

                to = to[bar]
                nrad = nrad[bar]
            '''

            if j == 0:
                xsecs = bt + to
                xnrad = nrad.T
            else:
                xsecs = np.append(xsecs, bt + to)
                xnrad = np.append(xnrad, nrad.T)

    nsecs = xsecs
    nrad = np.transpose(xnrad)

    # Quick check for consistency
    if len(nsecs) < len(secs):
        ni = len(secs) - len(nsecs)
        logger.debug(f"Adding summary samples: {ni}")
        nrad = np.append(nrad, np.array([nrad[-1, :] for k in range(ni)]), axis=0)

    elif len(nsecs) > len(secs):
        foo = np.where(nsecs <= max(secs)+1)
        if len(foo[0]) != len(secs):
            logger.error("Problem with time matching")
            return
        nsecs = nsecs[foo]
        nrad = nrad[foo, :]

    osecs = secs
    orad = rad.transpose()
    onrad = nrad.transpose()
    oqc = qcflag

    logger.info(f"Number of samples: {len(secs)}; Number of spectral channels: {len(wnum)}")

    # Select only the good data (simple check)
    if min(wnum) < 600:
        logger.debug("I believe I'm working with ch1 data")
        foo = np.where(wnum >= 900)
        minv = -2
        maxv = 105
    elif max(wnum) > 2800:
        logger.debug("I believe I'm working with ch2 data")
        foo = np.where(wnum >= 2550)
        minv = -0.2
        maxv = 2.0
    else:
        logger.error("Unable to determine the channel...")
        return

    good = np.where((minv <= rad[:, foo[0][0]]) & (rad[:, foo[0][0]] < maxv) & (oqc == 0))
    logger.debug(f"There are {len(good[0])} good samples out of {len(secs)}")

    # If the number of good spectra is too small, then we should abort
    if len(good[0]) <= 3*len(wnum):
        logger.critical("There are TOO FEW good spectra for a good PCA noise filter (need > 3x at least")
        logger.critical("Aborting")
        os.chdir(tdir)
        for fn in files:
            os.remove(os.path.basename(fn))
        return

    # If the keyword is set, use the median noise spectrum instead of the real noise spectrum
    if use_median_noise:
        logger.info("Using the median noise specturm, not the true sample noise spectrum")
        for i in range(nwnum):
            nrad[:, i] = np.median(nrad[:, i])

    # Normalize the data
    for i in range(len(secs)):
        rad[i, :] = rad[i, :] / np.interp(wnum, nwnum, nrad[i, :])

    # Make sure the pickle file exists
    if not os.path.exists(pcs_filename):
        logger.error(f"The PCS file {pcs_filename} does not exist!")
        os.chdir(tdir)
        for fn in files:
            os.remove(os.path.basename(fn))
        return

    # Open the pickle file and load the information 
    logger.info("Restoring PCs from a file (independent PCA noise filtering)")
    with open(pcs_filename, 'rb') as fh:
        data = pickle.load(fh)

    gsecs = data['gsecs']
    nvecs = data['nvecs']
    pcwnum = data['pcwnum']
    pca_comment = data['pca_comment']
    PC = data['PC']

    logger.debug(f"Number of e-vecs used in reconstruction: {nvecs}")

    # Confirm the wavenumber array here matches that used to generate the PCs
    delta = np.abs(wnum - pcwnum)
    foo = np.where(delta > 0.001)
    if len(foo[0]) > 0:
        logger.error("The wavenumber array with the PCs does not match the current wnum array")
        return

    # Project the data onto this reduced basis
    logger.debug('Projecting the coefficients')
    coef = pca_project(rad, nvecs, PC)

    # Expand the data
    logger.debug("Expanding the reduced basis set")
    frad = rad
    orad = rad
    feh = pca_expand(coef, PC)
    frad = feh

    # Remove the noise normalization
    for i in range(len(secs)):
        frad[i, :] = frad[i, :] * np.interp(wnum, nwnum, nrad[i, :])

    # Now place the noise-filtered data back into the netCDF files, and copy them to the output directory
    os.chdir(tdir)
    for fn in files:
        nc = Dataset(fn, 'a')
        rad = nc['mean_rad'][:]
        bt = nc['base_time'][0]
        
        if 'time_offset' in nc.variables.keys():
            to = nc['time_offset'][:]   
        elif 'time' in nc.variables.keys():
            bt *= 1e-3
            to = nc['time'][:]
        
        fsecs = bt+to
        kk = len(frad[0, :])

        for j, fsec in enumerate(fsecs):
            delta = np.abs(fsec - secs)
            foo = np.argmin(delta)
            if delta[foo] > 2:
                logger.error('Unable to find a sample I once had!')
                return

            rad[j, 0:kk] = frad[foo, :]

        nc['mean_rad'][:] = rad
        nc.setncattr('Noise_filter_comment', f'PCA noise filter was applied to the data, with {nvecs}'
                                             f' PCs used in the reconstruction')
        nc.setncattr('Noise_filter_comment2', pca_comment)
        nc.close()

        logger.debug(f"Final data in {os.path.join(odir, os.path.basename(fn))}")
        move(fn, os.path.join(odir, os.path.basename(fn)))














