# IRS Noise Filter

This code is intended to filter data from infrared spetrometers such as the Atmospheric Emitted Radiance Interferometer (AERI) or the Atmospheric Sounder Spectrometer by Infrared Spectral Technology (ASSIST). It is based on the algorithm published in Turner et al. (2006). 

The code can be used as a standalone Python package or run using the provided container depending on the end user needs. 

## Python
To install the Python package, simply clone this repository to the desired location. From inside the director, run:

```
pip install -r requirements.txt
pip install .
```

The package will then be importable via:

```
from irs_nf import irs_nf
```

The `run_irs_nf.py` script should be able to meet most user needs for filtering, though the main function can easily be integrated into other scripts. The inputs for `run_irs_nf.py` are found below

```
$ python run_irs_nf.py --help
usage: run_irs_nf.py [-h] [--pcs PCS] [--create] [--apply] [--tdir TDIR] [--verbose] start_date end_date idir sdir odir

positional arguments:
  start_date   Start date (YYYYmmdd)
  end_date     End date (YYYmmdd)
  idir         Directory with CH1 or CH2 data
  sdir         Directory with summary files
  odir         Output directory for the noise filtered files and the PCA file

optional arguments:
  -h, --help   show this help message and exit
  --pcs PCS    PCS file name
  --create     Flag to create a new PCA file
  --apply      Flag to apply the noise filter to the data
  --tdir TDIR  Temporary directory. If blank, one is created automatically
  --verbose    Verbose flag for debugging
```

Note that one or both of the `--create` and `--apply` flags need to be specified or the program will do nothing. The `--create` flag will result in a Python pickle file being written to the output directory (`odir`). The `--apply` flag will apply the filter to the data. 


## Docker/Podman

Coming soon!

### References

Turner, D. D., R. O. Knuteson, H. E. Revercomb, C. Lo, and R. G. Dedecker, 2006:
        Noise Reduction of Atmospheric Emitted Radiance Interferometer (AERI) Observations Using Principal Component Analysis. J. Atmos. Oceanic Technol., 23, 1223–1238. https://doi.org/10.1175/JTECH1906.1.
