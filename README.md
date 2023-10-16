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
usage: run_irs_nf.py [-h] [--create] [--apply] [--tdir TDIR] [--verbose] start_date end_date idir sdir odir {dmv2cdf,assist,arm}

positional arguments:
  start_date            Start date (YYYYmmdd)
  end_date              End date (YYYmmdd)
  idir                  Directory with CH1 or CH2 data
  sdir                  Directory with summary files
  odir                  Output directory for the noise filtered files and the PCA file
  {dmv2cdf,assist,arm}  Type of output files

optional arguments:
  -h, --help            show this help message and exit
  --create              Flag to create a new PCA file
  --apply               Flag to apply the noise filter to the data
  --tdir TDIR           Temporary directory. If blank, one is created automatically
  --verbose             Verbose flag for debugging
```

Note that one or both of the `--create` and `--apply` flags need to be specified or the program will do nothing. The `--create` flag will result in a Python pickle file being written to the output directory (`odir`). The `--apply` flag will apply the filter to the data. This allows the user to choose whether or not to perform independent or dependent noise filtering 


## Docker/Podman

This code is also packaged into a Docker image that can be pulled from [GitHub Packages](https://github.com/OAR-atmospheric-observations/IRS-Noise-Filter/pkgs/container/irs-noise-filter).

To run this you'll need to map the data location to the Docker container with the `-v` option in the run command. An example is shown below:

```
docker run --rm -v /Path/to/data/on/local/machine/:/data/ ghcr.io/oar-atmospheric-observations/irs-noise-filter:latest python run_irs_nf.py 20230820 20230830 /data/aeri /data/aeri /data/aeri_nf dmv2cdf --apply --create --verbose
```

### Disclamer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration (NOAA), or the United States Department of Commerce. All NOAA GitHub project code is provided on an ‘as is’ basis, with no warranty, and the user assumes responsibility for its use. NOAA has relinquished control of the information and no longer has responsibility to protect the integrity, confidentiality, or availability of the information. Any claims against the Department of Commerce or NOAA stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

### References

Turner, D. D., R. O. Knuteson, H. E. Revercomb, C. Lo, and R. G. Dedecker, 2006:
        Noise Reduction of Atmospheric Emitted Radiance Interferometer (AERI) Observations Using Principal Component Analysis. J. Atmos. Oceanic Technol., 23, 1223–1238. https://doi.org/10.1175/JTECH1906.1.
