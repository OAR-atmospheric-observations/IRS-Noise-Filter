import logging
import logging.config
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np

from irs_nf import irs_nf, utils

sfields = {
    'dmv2cdf': [['wnum1', 'SkyNENch1'], ['wnum2', 'SkyNENch2']],
    'assist': [['wnumsum1', 'SkyNENCh1'], ['wnumsum2', 'SkyNENCh2']],
    'arm': [['wnumsum5', 'SkyNENCh1'], ['wnumsum6', 'SkyNENCh2']]
}


parser = ArgumentParser()
parser.add_argument("start_date", action="store", help="Start date (YYYYmmdd)")
parser.add_argument("end_date", action="store", help="End date (YYYmmdd)")
parser.add_argument("idir", action="store", default=None, help="Directory with CH1 or CH2 data")
parser.add_argument("sdir", action="store", help="Directory with summary files")
parser.add_argument("odir", action="store", help="Output directory for the noise filtered files and the PCA file")
parser.add_argument("irs_type", action='store', choices=sfields.keys(), help="Type of output files")
# parser.add_argument("--pcs", required=False, action="store", help="PCS file name")
parser.add_argument("--create", default=False, action="store_true", help="Flag to create a new PCA file")
parser.add_argument("--apply", default=False, action="store_true", help="Flag to apply the noise filter to the data")
parser.add_argument("--tdir", required=False, action="store", help="Temporary directory. If blank, one is created automatically")
parser.add_argument("--verbose", required=False, action='store_true', default=False, help="Verbose flag for debugging")
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(format='%(asctime)s | %(levelname)s -> %(message)s', level=logging.DEBUG, datefmt="%Y-%m-%d-%H:%M:%S")
else:
    logging.basicConfig(format='%(asctime)s | %(levelname)s -> %(message)s', level=logging.INFO, datefmt="%Y-%m-%d-%H:%M:%S")

logger = logging.getLogger("irs_logger")
logger.info("Starting IRS PCA Noise Filter")

# Extract the args to variables
start = datetime.strptime(args.start_date, "%Y%m%d")
end = datetime.strptime(args.end_date, "%Y%m%d")
idir = args.idir
sdir = args.sdir
odir = args.odir
tdir = args.tdir
create_pca = args.create
apply_nf = args.apply

if (create_pca is False) & (apply_nf is False):
    logger.fatal("Neither --create nor --apply were specified in the arguments, thus this code will do nothing.")
    sys.exit(0)

# TODO - Need to make this more generic, still...
ch1_sfield = sfields[args.irs_type][0]
ch2_sfield = sfields[args.irs_type][1]

if args.irs_type == "assist":
    sky_view_angle = 0
else:
    sky_view_angle = None

# If the directories we write to aren't there, make them
if not os.path.exists(odir):
    os.makedirs(odir)

# Make a temp directory if needed
if tdir is None:
    tdir = os.path.join(odir, 'tmp')

if not os.path.exists(tdir):
    os.makedirs(tdir)

# Show the info to the user
logger.info(f"Running the noise filter between {start:%Y%m%d} and {end:%Y%m%d}")
logger.debug(f"Input dir: {idir}")
logger.debug(f"Summary dir: {sdir}")
logger.debug(f"Output dir: {odir}")
logger.debug(f"Temporary dir: {tdir}")
logger.debug(f"PCS Filename: {tdir}")

# Get the files needed for channel 1
os.chdir(idir)
files, err = utils.findfile(idir, "*(ch1|cha|ChA)*.(nc|cdf)")
rdates = np.array([datetime.strptime(d.split('.')[-3], '%Y%m%d') for d in files])
foo = np.where((rdates >= start) & (rdates <= end))

rfilenames = np.array(files)[foo]

logger.info(f"Number of Ch1 files: {len(rfilenames)}")

if len(rfilenames) >= 1:

    pcs_filename = os.path.join(odir, 'irs_nf_ch1.pkl')
    
    if create_pca:
        logger.info(f"Creating IRS filter (Ch1)")
        logger.info(f"Storing in {pcs_filename}")
        irs_nf.create_irs_noise_filter(rfilenames, sdir, ch1_sfield,  
                                       pcs_filename=pcs_filename, sky_view_angle=sky_view_angle)

    if apply_nf:
        logger.info("Applying IRS filter (Ch1)")
        irs_nf.apply_irs_noise_filter(rfilenames, sdir, ch1_sfield, tdir, odir, 
                                      pcs_filename=pcs_filename)
logger.info("Finished with Ch1")
    
# Get the files needed for channel 2
os.chdir(idir)
files, err = utils.findfile(idir, "*(ch2|chb|ChB)*.(nc|cdf)")
rdates = np.array([datetime.strptime(d.split('.')[-3], '%Y%m%d') for d in files])
foo = np.where((rdates >= start) & (rdates <= end))

rfilenames = np.array(files)[foo]

logger.info(f"Number of Ch2 files: {len(rfilenames)}")

if len(rfilenames) >= 1:

    pcs_filename = os.path.join(odir, 'irs_nf_ch2.pkl')
    
    if create_pca:
        logger.info(f"Creating IRS filter (Ch2)")
        logger.info(f"Storing in {pcs_filename}")
        irs_nf.create_irs_noise_filter(rfilenames, sdir, ch2_sfield,  
                                       pcs_filename=pcs_filename, sky_view_angle=sky_view_angle)

    if apply_nf:
        logger.info("Applying IRS filter (Ch2)")
        irs_nf.apply_irs_noise_filter(rfilenames, sdir, ch2_sfield, tdir, odir, 
                                      pcs_filename=pcs_filename)
logger.info("Finished with Ch2")