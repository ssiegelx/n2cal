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


class TransitTracker(object):

    def __init__(self, nsigma=3.0, extend_night=1800.0):

        self._entries = {}
        self._nsigma = nsigma
        self._extend_night = extend_night

    def add_file(self, filename):

        # Make sure this file is not currently in the transit tracker
        if self.contains_file(filename):
            return

        # Read file time range
        with h5py.File(filename, 'r') as handler:
            timestamp = handler['index_map']['time']['ctime'][:]

        timestamp0 = np.median(timestamp)

        # Convert to right ascension
        ra = ephemeris.lsa(timestamp)
        csd = ephemeris.csd(timestamp)

        # Loop over available sources
        for name, src in self.iteritems():

            src_ra, src_dec = ephemeris.object_coords(src.body, date=timestamp0, deg=True)

            # Determine if any times in this file fall
            # in a window around transit of this source
            hour_angle = ra - src_ra
            hour_angle = hour_angle + 360.0 * (hour_angle < -180.0) - 360.0 * (hour_angle > 180.0)

            good_time = np.flatnonzero(np.abs(hour_angle) < src.window)

            if good_time.size > 0:

                # Determine the csd for the transit contained in this file
                icsd = np.unique(np.floor(csd[good_time] - (hour_angle[good_time] / 360.0)))

                if icsd.size > 1:
                    RuntimeError("Error estimating CSD.")

                key = int(icsd[0])

                min_ha, max_ha = np.percentile(hour_angle, [0, 100])

                # Add to list of files to analyze for this source
                if key in src.files:
                    src.files[key].append((filename, hour_angle))
                    src.file_span[key][0] = min(min_ha, src.file_span[key][0])
                    src.file_span[key][1] = max(max_ha, src.file_span[key][1])

                else:
                    src.files[key] = [(filename, hour_angle)]
                    src.file_span[key] = [min_ha, max_ha]

    def get_transits(self):

        out = []
        for name, src in self.iteritems():

            for csd in sorted(src.file_span.keys()):

                span = src.file_span[csd]

                #if (span[0] <= -src.window) and (span[1] >= src.window):

                files = src.files.pop(csd)

                isort = np.argsort([np.min(ha) for ff, ha in files])

                hour_angle = np.concatenate(tuple([files[ii][1] for ii in isort]))

                if np.all(np.diff(hour_angle) > 0.0):

                    below = np.flatnonzero(hour_angle <= -src.window)
                    aa = int(np.max(below)) if below.size > 0 else 0

                    above = np.flatnonzero(hour_angle >=  src.window)
                    bb = int(np.min(above)) if above.size > 0 else hour_angle.size

                    is_day = self.is_daytime(src, csd)

                    out.append((name, csd, is_day, [files[ii][0] for ii in isort],  aa, bb))

                del src.file_span[csd]

        return out

    def is_daytime(self, key, csd):

        src = self[key] if isinstance(key, basestring) else key

        is_daytime = 0

        src_ra, src_dec = ephemeris.object_coords(src.body, date=ephemeris.csd_to_unix(csd), deg=True)

        transit_start = ephemeris.csd_to_unix(csd + (src_ra - src.window) / 360.0)
        transit_end = ephemeris.csd_to_unix(csd + (src_ra + src.window) / 360.0)

        solar_rise = ephemeris.solar_rising(transit_start - 24.0*3600.0, end_time=transit_end)

        for rr in solar_rise:

            ss = ephemeris.solar_setting(rr)[0]

            rrex = rr + self._extend_night
            ssex = ss - self._extend_night

            if ((transit_start <= ssex) and (rrex <= transit_end)):

                is_daytime += 1

                tt = ephemeris.solar_transit(rr)[0]
                if (transit_start <= tt) and (tt <= transit_end):
                    is_daytime += 1

                break

        return is_daytime

    def contains_file(self, filename):

        contains = False
        for name, src in self.iteritems():
            for csd, file_list in src.files.iteritems():

                if filename in [ff[0] for ff in file_list]:

                    contains = True

        return contains

    def __setitem__(self, key, body):

        if key not in self:

            if ephemeris._is_skyfield_obj(body):
                pass
            elif isinstance(body, (tuple, list)) and (len(body) == 2):
                ra, dec = body
                body = ephemeris.skyfield_star_from_ra_dec(ra, dec, bd_name=key)
            else:
                ValueError("Item must be skyfield object or tuple (ra, dec).")

            #window = self._nsigma * cal_utils.guess_fwhm(400.0, pol='X', dec=body.dec.radians, sigma=True)
            window = self._nsigma * utils.get_window(400.0, pol='X', dec=body.dec.radians, deg=True)

            self._entries[key] = NameSpace()
            self._entries[key].body = body
            self._entries[key].window = window
            self._entries[key].files = {}
            self._entries[key].file_span = {}

    def __contains__(self, key):

        return key in self._entries

    def __getitem__(self, key):

        if key not in self:
            raise KeyError

        return self._entries[key]

    def iteritems(self):

        return self._entries.iteritems()

    def clear_all(self):

        self._entries = {}

    def remove(self, key):

        self._entries.pop(key)

###################################################
# main routine
###################################################

# def main(acq="20180327T022127Z_chimeN2_corr",  output_dir="/home/ssiegel/chime/eigen",
#                                                all_sources=None,
#                                                niceness=10,  nsigma=2.0,
#                                                freq_start=None, freq_stop=None,
#                                                neigen=16, niter=10, offsets=None,
#                                                cross_pol=True, sep_pol=False,
#                                                normalize=True, good_csd=None):

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

    # Create transit tracker
    transit_tracker = TransitTracker(nsigma=config.nsigma)

    for name in config.all_sources:
        transit_tracker[name] = FluxCatalog[name].skyfield

    for aa in acq_files:
        transit_tracker.add_file(aa)

    transit_files = transit_tracker.get_transits()

    for src, csd, is_day, files, aa, bb in transit_files:
        mlog.info("%s | CSD %d | %d | (%d, %d)" % (src, csd, is_day, aa, bb))
        for ff in files:
            mlog.info("%s" % ff)

    mlog.info(''.join(['-'] * 80))

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
        rank = 2
        cross_pol = config.cross_pol
        pol = np.array(['all'])

    npol = pol.size

    # Create file suffix
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

    # Loop over source transits
    for src, csd, is_day, files, start, stop in transit_files:

        if src not in config.all_sources:
            continue

        if (config.good_csd is not None) and (csd not in config.good_csd):
            continue

        mlog.info("%s | CSD %d | %d | (%d, %d)" % (src, csd, is_day, start, stop))

        nfiles = len(files)

        output_file = os.path.join(config.output_dir, "gain_solutions_%s_CSD_%d_%s.pickle"  % (src, csd, suffix))

        mlog.info("Saving to:  %s" % output_file)

        # Get info about this set of files
        data = andata.CorrData.from_acq_h5(files, datasets=['flags/inputs'], start=start, stop=stop)

        ra = ephemeris.lsa(data.time)
        prod = data.prod

        fstart = config.freq_start if config.freq_start is not None else 0
        fstop = config.freq_stop if config.freq_stop is not None else data.freq.size
        freq_index = range(fstart, fstop)

        freq =  data.freq[freq_index]

        ntime = ra.size
        nfreq = freq.size

        # Determind bad inputs
        if config.bad_input_file is None or not os.path.isfile(config.bad_input_file):
            bad_input = np.flatnonzero(~np.all(data.flags['inputs'][:], axis=-1))
        else:
            with open(config.bad_input_file, 'r') as handler:
                bad_input = pickle.load(handler)

        mlog.info("%d inputs flagged as bad." % bad_input.size)
        bad_prod = np.array([ii for ii, pp in enumerate(prod) if (pp[0] in bad_input) or (pp[1] in bad_input)])

        # Determine time range of each file
        findex = []
        tindex = []
        for ii, filename in enumerate(files):
            subdata = andata.CorrData.from_acq_h5(filename, datasets=())

            findex += [ii] * subdata.ntime
            tindex += range(subdata.ntime)

        findex = np.array(findex[start:stop])
        tindex = np.array(tindex[start:stop])

        frng = []
        for ii in range(nfiles):

            this_file = np.flatnonzero(findex == ii)
            this_tindex = tindex[this_file]

            frng.append((this_tindex.min(), this_tindex.max()+1))

        # Create arrays to hold the results
        ores = {}
        ores['src'] = src
        ores['csd'] = csd
        ores['ra'] = ra
        ores['time'] = data.time
        ores['freq'] = freq
        ores['offsets'] = config.offsets
        ores['pol'] = pol

        ores['evalue'] = np.zeros((noffset, nfreq, ntime, N), dtype=np.float32)
        ores['resp'] = np.zeros((noffset, nfreq, ntime, N, config.neigen),  dtype=np.complex64)
        ores['resp_err'] = np.zeros((noffset, nfreq, ntime, N, config.neigen), dtype=np.float32)

        # Loop over frequencies
        for ff, find in enumerate(freq_index):

            mlog.info("Freq %d of %d." % (ff+1, nfreq))

            cnt = 0

            # Loop over files
            for ii, filename in enumerate(files):

                aa, bb = frng[ii]

                # Loop over times
                for tt in range(aa, bb):

                    t0 = time.time()

                    mlog.info("Time %d of %d." % (cnt+1, ntime))

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

        # Remove this source from list
        config.all_sources.remove(src)


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