import os
import sys
from glob import glob
import pickle
import argparse
import time

import numpy as np
import h5py
import scipy.linalg as la

from pychfpga import NameSpace, load_yaml_config
from calibration import utils

import log

import caput.time as ctime
import time
import skyfield.api

from ch_util import tools, ephemeris, andata
from ch_util.fluxcat import FluxCatalog

sys.path.insert(0, "/home/ssiegel/ch_pipeline/venv/src/draco")
from draco.util import _fast_tools

###################################################
# default variables
###################################################

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':n2cal'))

LOG_FILE = os.environ.get('N2CAL_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'n2cal.log'))

DEFAULT_LOGGING = {
    'formatters': {
         'std': {
             'format': "%(asctime)s %(levelname)s %(name)s: %(message)s",
             'datefmt': "%m/%d %H:%M:%S"},
          },
    'handlers': {
        'stderr': {'class': 'logging.StreamHandler', 'formatter': 'std', 'level': 'DEBUG'}
        },
    'loggers': {
        '': {'handlers': ['stderr'], 'level': 'INFO'}  # root logger

        }
    }

###################################################
# ancillary routines
###################################################

def coupled_indices(cutoff=0, cross_pol=True, N=2048, cyl_size=256):

    Ncyl = N / cyl_size

    row = []
    col = []

    for cc in range(Ncyl):

        acyl = cc*cyl_size
        bcyl = (cc+1)*cyl_size

        for ii in range(acyl, bcyl):

            aa = max(acyl, ii - cutoff)
            bb = min(bcyl, ii + cutoff + 1)

            for jj in range(aa, bb):

                row.append(ii)
                col.append(jj)


    if cross_pol:

        for cc in range(Ncyl):

            acyl = cc*cyl_size
            bcyl = (cc+1)*cyl_size

            offset = cyl_size - 2*cyl_size*(cc %  2)
            across = offset + acyl
            bcross = offset + bcyl

            for ii in range(acyl, bcyl):

                aa = max(across, offset + ii - cutoff)
                bb = min(bcross, offset + ii + cutoff + 1)

                for jj in range(aa, bb):

                    row.append(ii)
                    col.append(jj)

    return (np.array(row), np.array(col))


def rankN_approx(A, rank=1):
    """Create the rank-N approximation to the matrix A.

    Parameters
    ----------
    A : np.ndarray
        Matrix to approximate
    rank : int, optional

    Returns
    -------
    B : np.ndarray
        Low rank approximation.
    """

    N = A.shape[0]

    evals, evecs = la.eigh(A, eigvals=(N - rank, N - 1))

    return np.dot(evecs, evals[:, np.newaxis] * evecs.T.conj())


def eigh_no_diagonal(A, niter=5, rank=1, diag_index=None, eigvals=None):
    """Eigenvalue decomposition ignoring the diagonal elements.

    The diagonal elements are iteratively replaced with those from a rank=1 approximation.

    Parameters
    ----------
    A : np.ndarray[:, :]
        Matrix to decompose.
    niter : int, optional
        Number of iterations to perform.
    eigvals : (lo, hi), optional
        Indices of eigenvalues to select (inclusive).

    Returns
    -------
    evals : np.ndarray[:]
    evecs : np.ndarray[:, :]
    """

    Ac = A.copy()

    if niter > 0:

        if diag_index is None:
            diag_index  = np.diag_indices(Ac.shape[0])

        Ac[diag_index] = 0.0

        for i in range(niter):
            Ac[diag_index] = rankN_approx(Ac, rank=rank)[diag_index]

    return la.eigh(Ac, eigvals=eigvals)


def _extract_diagonal(utmat, axis=1):
    """Extract the diagonal elements of an upper triangular array.

    Parameters
    ----------
    utmat : np.ndarray[..., nprod, ...]
        Upper triangular array.
    axis : int, optional
        Axis of array that is upper triangular.

    Returns
    -------
    diag : np.ndarray[..., ninput, ...]
        Diagonal of the array.
    """

    # Estimate nside from the array shape
    nside = int((2 * utmat.shape[axis])**0.5)

    # Check that this nside is correct
    if utmat.shape[axis] != (nside * (nside + 1) / 2):
        msg = ('Array length (%i) of axis %i does not correspond upper triangle\
                of square matrix' % (utmat.shape[axis], axis))
        raise RuntimeError(msg)

    # Find indices of the diagonal
    diag_ind = [tools.cmap(ii, ii, nside) for ii in range(nside)]

    # Construct slice objects representing the axes before and after the product axis
    slice0 = (np.s_[:],) * axis
    slice1 = (np.s_[:],) * (len(utmat.shape) - axis - 1)

    # Extract wanted elements with a giant slice
    sl = slice0 + (diag_ind,) + slice1
    diag_array = utmat[sl]

    return diag_array


def solve_gain(data, cutoff=0, cross_pol=True, normalize=True, rank=1, niter=5, neigen=1):

    # Turn into numpy array to avoid any unfortunate indexing issues
    data = data[:].view(np.ndarray)

    # Calcuate the number of feeds in the data matrix
    tfeed = int((2 * data.shape[0])**0.5)

    # If not set, create the list of included feeds (i.e. all feeds)
    feeds = np.arange(tfeed)
    nfeed = len(feeds)

    # Create empty arrays to store the outputs
    gain = np.zeros((nfeed, neigen), np.complex64)
    gain_error = np.zeros((nfeed, neigen), np.float32)

    # Set up normalisation matrix
    auto = (_extract_diagonal(data, axis=0).real)**0.5

    if normalize:
        inv_norm = auto
    else:
        inv_norm = np.ones_like(auto)

    norm = tools.invert_no_zero(inv_norm)

    # Initialise a temporary array for unpacked products
    cd = np.zeros((nfeed, nfeed), dtype=data.dtype)

    # Unpack visibility and normalisation array into square matrix
    _fast_tools._unpack_product_array_fast(data[:].copy(), cd, feeds, tfeed)

    # Apply weighting
    w = norm
    cd *= np.outer(norm, norm.conj())

    # Skip if any non-finite values
    if not np.isfinite(cd).all():
        raise ValueError

    # Compute diag indices
    diag_index = coupled_indices(cutoff=cutoff, cross_pol=cross_pol, N=nfeed)

    # Solve for eigenvectors and eigenvalues
    evals, evecs = eigh_no_diagonal(cd, rank=rank, niter=niter, diag_index=diag_index)

    # Construct gain solutions
    if evals[-1] > 0:

        for ki in range(neigen):

            kk = -1 - ki

            sign0 = (1.0 - 2.0 * (evecs[0, kk].real < 0.0))

            gain[:, ki] = sign0 * inv_norm * evecs[:, kk] * evals[kk]**0.5

            gain_error[:, ki] = (inv_norm * np.median(np.abs(evals[:-2] - np.median(evals[:-2]))) /
                                (nfeed * evals[kk])**0.5)

    # If neigen = 1, remove single dimension
    if neigen == 1:
        gain = np.squeeze(gain, axis=-1)
        gain_error = np.squeeze(gain_error, axis=-1)


    return evals, gain, gain_error

def _correct_phase_wrap(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi

def sun_coord(unix_time, deg=True):

    date = ephemeris.ensure_unix(np.atleast_1d(unix_time))
    skyfield_time = ephemeris.unix_to_skyfield_time(date)
    ntime = date.size

    coord = np.zeros((ntime, 4), dtype=np.float32)

    planets = skyfield.api.load('de421.bsp')
    sun = planets['sun']

    observer = ephemeris._get_chime().skyfield_obs()

    apparent = observer.at(skyfield_time).observe(sun).apparent()
    radec = apparent.radec(epoch=skyfield_time)

    coord[:, 0] = radec[0].radians
    coord[:, 1] = radec[1].radians

    altaz = apparent.altaz()
    coord[:, 2] = altaz[0].radians
    coord[:, 3] = altaz[1].radians

    # Correct RA from equinox to CIRS coords using
    # the equation of the origins
    era = np.radians(ctime.unix_to_era(date))
    gast = 2 * np.pi * skyfield_time.gast / 24.0
    coord[:, 0] = coord[:, 0] + (era - gast)

    # Convert to hour angle
    coord[:, 0] = _correct_phase_wrap(coord[:, 0] - np.radians(ephemeris.lsa(date)))

    if deg:
        coord = np.degrees(coord)

    return coord

###################################################
# main routine
###################################################

def main(config_file=None, logging_params=DEFAULT_LOGGING):

    # Setup logging
    log.setup_logging(logging_params)
    mlog = log.get_logger(__name__)

    # Set config
    config = DEFAULTS.deepcopy()
    if config_file is not None:
        config.merge(NameSpace(load_yaml_config(config_file)))

    # Set niceness
    current_niceness = os.nice(0)
    os.nice(config.niceness - current_niceness)
    mlog.info('Changing process niceness from %d to %d.  Confirm:  %d' %
                  (current_niceness, config.niceness, os.nice(0)))

    # Find acquisition files
    acq_files = sorted(glob(os.path.join(config.data_dir, config.acq, "*.h5")))
    nfiles = len(acq_files)

    # Determine time range of each file
    findex = []
    tindex = []
    for ii, filename in enumerate(acq_files):
        subdata = andata.CorrData.from_acq_h5(filename, datasets=())

        findex += [ii] * subdata.ntime
        tindex += range(subdata.ntime)

    findex = np.array(findex)
    tindex = np.array(tindex)

    # Determine transits within these files
    transits = []

    data = andata.CorrData.from_acq_h5(acq_files, datasets=())

    solar_rise = ephemeris.solar_rising(data.time[0] - 24.0 * 3600.0, end_time=data.time[-1])

    for rr in solar_rise:

        ss = ephemeris.solar_setting(rr)[0]

        solar_flag = np.flatnonzero((data.time >= rr) & (data.time <= ss))

        if solar_flag.size > 0:

            solar_flag = solar_flag[::config.downsample]

            tval = data.time[solar_flag]

            this_findex = findex[solar_flag]
            this_tindex = tindex[solar_flag]

            file_list, tindices = [], []

            for ii in range(nfiles):

                this_file = np.flatnonzero(this_findex == ii)

                if this_file.size > 0:

                    file_list.append(acq_files[ii])
                    tindices.append(this_tindex[this_file])

            date = ephemeris.unix_to_datetime(rr).strftime('%Y%m%dT%H%M%SZ')
            transits.append((date, tval, file_list, tindices))

    # Specify some parameters for algorithm
    N = 2048

    noffset = len(config.offsets)

    if config.sep_pol:
        rank = 1
        cross_pol = False
        pol = np.array(['S', 'E'])
        pol_s = np.array([rr + 256*xx for xx in range(0, 8, 2) for rr in range(256)])
        pol_e = np.array([rr + 256*xx for xx in range(1, 8, 2) for rr in range(256)])
        prod_ss = []
        prod_ee = []
    else:
        rank = 8
        cross_pol = config.cross_pol
        pol = np.array(['all'])

    npol = pol.size

    # Create file prefix and suffix
    prefix = []

    prefix.append("gain_solutions")

    if config.output_prefix is not None:
        prefix.append(config.output_prefix)

    prefix = '_'.join(prefix)


    suffix = []

    suffix.append("pol_%s" % '_'.join(pol))

    suffix.append("niter_%d" % config.niter)

    if cross_pol:
        suffix.append("zerocross")
    else:
        suffix.append("keepcross")

    if config.normalize:
        suffix.append("normed")
    else:
        suffix.append("notnormed")

    suffix = '_'.join(suffix)

    # Loop over solar transits
    for date, timestamps, files, time_indices in transits:

        nfiles = len(files)

        mlog.info("%s (%d files) " % (date, nfiles))

        output_file = os.path.join(config.output_dir, "%s_SUN_%s_%s.pickle"  % (prefix, date, suffix))

        mlog.info("Saving to:  %s" % output_file)

        # Get info about this set of files
        data = andata.CorrData.from_acq_h5(files, datasets=['flags/inputs'])

        prod = data.prod

        coord = sun_coord(timestamps, deg=True)

        fstart = config.freq_start if config.freq_start is not None else 0
        fstop = config.freq_stop if config.freq_stop is not None else data.freq.size
        freq_index = range(fstart, fstop)

        freq =  data.freq[freq_index]

        ntime = timestamps.size
        nfreq = freq.size

        # Determind bad inputs
        if config.bad_input_file is None or not os.path.isfile(config.bad_input_file):
            bad_input = np.flatnonzero(~np.all(data.flags['inputs'][:], axis=-1))
        else:
            with open(config.bad_input_file, 'r') as handler:
                bad_input = pickle.load(handler)

        mlog.info("%d inputs flagged as bad." % bad_input.size)
        bad_prod = np.array([ii for ii, pp in enumerate(prod) if (pp[0] in bad_input) or (pp[1] in bad_input)])

        # Create arrays to hold the results
        ores = {}
        ores['date'] = date
        ores['coord'] = coord
        ores['time'] = timestamps
        ores['freq'] = freq
        ores['offsets'] = config.offsets
        ores['pol'] = pol

        ores['evalue'] = np.zeros((noffset, nfreq, ntime, N), dtype=np.float32)
        ores['resp'] = np.zeros((noffset, nfreq, ntime, N, config.neigen),  dtype=np.complex64)
        ores['resp_err'] = np.zeros((noffset, nfreq, ntime, N, config.neigen), dtype=np.float32)

        # Loop over frequencies
        for ff, find in enumerate(freq_index):

            mlog.info("Freq %d of %d.  %0.2f MHz." % (ff+1, nfreq, freq[ff]))

            cnt = 0

            # Loop over files
            for ii, (filename, tind) in enumerate(zip(files, time_indices)):

                ntind = len(tind)
                mlog.info("Processing file %s (%d time samples)" % (filename, ntind))

                # Loop over times
                for tt in tind:

                    t0 = time.time()

                    mlog.info("Time %d of %d.  %d index of current file." % (cnt+1, ntime, tt))

                    # Load visibilities
                    with h5py.File(filename, 'r') as hf:

                        vis = hf['vis'][find, :, tt]

                    # Set bad products equal to zero
                    vis[bad_prod] = 0.0

                    # Different code if we are separating polarisations
                    if config.sep_pol:

                        if not any(prod_ss):

                            for pind, pp in enumerate(prod):
                                if (pp[0] in pol_s) and (pp[1] in pol_s):
                                    prod_ss.append(pind)

                                elif (pp[0] in pol_e) and (pp[1] in pol_e):
                                    prod_ee.append(pind)

                            prod_ss = np.array(prod_ss)
                            prod_ee = np.array(prod_ee)

                            mlog.info("Product sizes: %d, %d" % (prod_ss.size, prod_ee.size))

                        # Loop over polarisations
                        for pp, (input_pol, prod_pol) in enumerate([(pol_s, prod_ss), (pol_e, prod_ee)]):

                            visp = vis[prod_pol]

                            mlog.info("pol %s, visibility size:  %d" % (pol[pp], visp.size))

                            # Loop over offsets
                            for oo, off in enumerate(config.offsets):

                                mlog.info("pol %s, rank %d, niter %d, offset %d, cross_pol %s, neigen %d" % (pol[pp], rank, config.niter, off, cross_pol, config.neigen))

                                ev, rr, rre = solve_gain(visp, cutoff=off, cross_pol=cross_pol, normalize=config.normalize,
                                                               rank=rank, niter=config.niter, neigen=config.neigen)

                                ores['evalue'][oo, ff, cnt, input_pol] = ev
                                ores['resp'][oo, ff, cnt, input_pol, :] = rr
                                ores['resp_err'][oo, ff, cnt, input_pol, :] = rre

                    else:

                        # Loop over offsets
                        for oo, off in enumerate(config.offsets):

                            mlog.info("rank %d, niter %d, offset %d, cross_pol %s, neigen %d" % (rank, config.niter, off, cross_pol, config.neigen))

                            ev, rr, rre = solve_gain(vis, cutoff=off, cross_pol=cross_pol, normalize=config.normalize,
                                                          rank=rank, niter=config.niter, neigen=config.neigen)

                            ores['evalue'][oo, ff, cnt, :] = ev
                            ores['resp'][oo, ff, cnt, :, :] = rr
                            ores['resp_err'][oo, ff,  cnt, :, :] = rre

                    # Increment time counter
                    cnt += 1

                    # Print time elapsed
                    mlog.info("Took %0.1f seconds." % (time.time() - t0,))


        # Save to pickle file
        with open(output_file, 'w') as handle:

            pickle.dump(ores, handle)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config',   help='Name of configuration file.',
                                      type=str, default=None)
    parser.add_argument('--log',      help='Name of log file.',
                                      type=str, default=LOG_FILE)

    args = parser.parse_args()

    # If calling from the command line, then send logging to log file instead of screen
    try:
        os.makedirs(os.path.dirname(args.log))
    except OSError:
        if not os.path.isdir(os.path.dirname(args.log)):
            raise

    logging_params = DEFAULT_LOGGING
    logging_params['handlers'] = {'stderr': {'class': 'logging.handlers.WatchedFileHandler',
                                             'filename': args.log, 'formatter': 'std', 'level': 'INFO'}}

    # Call main routine
    main(config_file=args.config, logging_params=logging_params)
