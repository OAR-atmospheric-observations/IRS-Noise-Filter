from distutils.core import setup

setup(name='IRS Noise Filter',
      version='1.0',
      description='PCA noise filter for use with infrared spectrometer such as the AERI and ASSIST',
      author='Tyler M. Bell',
      author_email='tyler.bell@noaa.gov',
      packages=['irs_nf'],
      requires=['numpy', 'scipy', 'matplotlib', 'netCDF4'],
      scripts=['./run_irs_nf.py']
     )