import logging
import logging.config
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np

from irs_nf import irs_nf, utils

parser = ArgumentParser()
parser.add_argument("start_date", action="store", help="Start date (YYYYmmdd)")
parser.add_argument("end_date", action="store", help="End date (YYYmmdd)")
parser.add_argument("idir", action="store", default=None, help="Directory with CH1 or CH2 data")
parser.add_argument("sdir", action="store", help="Directory with summary files")
parser.add_argument("odir", action="store", help="Output directory for the noise filtered files and the PCA file")
parser.add_argument("--pcs", required=False, action="store", help="PCS file name")
parser.add_argument("--create", default=False, action="store_true", help="Flag to create a new PCA file")
parser.add_argument("--apply", default=False, action="store_true", help="Flag to apply the noise filter to the data")
parser.add_argument("--tdir", required=False, action="store", help="Temporary directory. If blank, one is created automatically")
parser.add_argument("--verbose", required=False, action='store_true', default=False, help="Verbose flag for debugging")
args = parser.parse_args()


if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

log = logging.getLogger("irs_logger")
log.info("Starting IRS PCA Noise Filter")

# Extract the args to variables
start = datetime.strptime(args.start_date, "%Y%m%d")
end = datetime.strptime(args.end_date, "%Y%m%d")
idir = args.idir
sdir = args.sdir
odir = args.odir
tdir = args.tdir
create_pca = args.create
apply_nf = args.apply
pcs_filename = args.pcs

if pcs_filename is None:
    pcs_filename = f"{odir}/nf_pcs.pkl"

if (create_pca is False) & (apply_nf is False):
    log.fatal("Neither --create nor --apply were specified in the arguments, thus this code will do nothing.")
    sys.exit(0)

# TODO - Need to make this more generic...
# Set the variable names to look at in the summary file
# For DMV files, ARM files are ['wnumsum5','SkyNENCh1']
sfields = ['wnum1', 'SkyNENch1']    

# If the directories we write to aren't there, make them
if not os.path.exists(odir):
    os.makedirs(odir)

# Make a temp directory if needed
if tdir is None:
    tdir = os.path.join(odir, 'tmp')

if not os.path.exists(tdir):
    os.makedirs(tdir)

# Show the info to the user
log.info(f"Running the noise filter between {start:%Y%m%d} and {end:%Y%m%d}")
log.debug(f"Input dir: {idir}")
log.debug(f"Summary dir: {sdir}")
log.debug(f"Output dir: {odir}")
log.debug(f"Temporary dir: {tdir}")
log.debug(f"PCS Filename: {tdir}")

# Get the files needed
os.chdir(idir)
files, err = utils.findfile(idir, "*(assist|aeri)*.(nc|cdf)")
rdates = np.array([datetime.strptime(d.split('.')[-3], '%Y%m%d') for d in files])
foo = np.where((rdates >= start) & (rdates <= end))

rfilenames = np.array(files)[foo]

log.debug(f"Number of files: {len(rfilenames)}")

if create_pca:
    irs_nf.irs_noise_filter(idir, sdir, sfields, tdir, odir,
                                  rfilenames, create_pcs_only=True, pcs_filename=pcs_filename)

if apply_nf:
    irs_nf.irs_noise_filter(idir, sdir, sfields, tdir, odir,
                              rfilenames, apply_only=True, pcs_filename=pcs_filename)