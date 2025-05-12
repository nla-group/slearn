import setuptools
import platform
import importlib
import logging

PACKAGE_NAME = "slearn"
VERSION = "0.2.6"
SETUP_REQUIRES = ["numpy>=1.17.2"] 
INSTALL_REQUIRES = [  
    "numpy>=1.17.2",
    "scikit-learn",
    "pandas",
    "lightgbm",
    "requests",
    "textdistance"
]
MAINTAINER = "nla-group"
EMAIL = "stefan.guettel@manchester.ac.uk"
AUTHORS = "Roberto Cahuantzi, Xinye Chen, Stefan Güttel"

with open("README.rst", 'r') as f:
    long_description = f.read()

ext_errors = (ModuleNotFoundError, IOError, SystemExit)
logging.basicConfig()
log = logging.getLogger(__file__)

if platform.python_implementation() == "PyPy":
    NUMPY_MIN_VERSION = "1.19.2"
else:
    NUMPY_MIN_VERSION = "1.17.2"

metadata = {
    "name": PACKAGE_NAME,
    "packages": [PACKAGE_NAME],
    "version": VERSION,
    "setup_requires": SETUP_REQUIRES,
    "install_requires": INSTALL_REQUIRES,
    # Defer numpy.get_include() to avoid requiring numpy at module level
    "include_dirs": [],  # Will be set in setup_package
    "long_description": long_description,
    "author": AUTHORS,
    "maintainer": MAINTAINER,
    "author_email": EMAIL,
    "classifiers": [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
    ],
    "maintainer_email": EMAIL,
    "description": "A package linking symbolic representation with sklearn for time series prediction",
    "long_description_content_type": 'text/x-rst',
    "url": "https://github.com/nla-group/slearn.git",
    "license": 'MIT License'
}

class InvalidVersion(ValueError):
    """Raise invalid version error"""

def check_package_status(package, min_version):
    """
    Check whether given package is installed and meets the minimum version.
    """
    import traceback
    package_status = {}
    try:
        module = importlib.import_module(package)
        package_version = module.__version__
        package_status["up_to_date"] = package_version >= min_version
        package_status["version"] = package_version
    except ImportError:
        traceback.print_exc()
        package_status["up_to_date"] = False
        package_status["version"] = ""

    req_str = f"slearn requires {package} >= {min_version}.\n"

    if not package_status["up_to_date"]:
        if package_status["version"]:
            raise ImportError(
                f"Your installation of {package} {package_status['version']} is out-of-date.\n{req_str}"
            )
        else:
            raise ImportError(
                f"{package} is not installed.\n{req_str}"
            )

def setup_package():
    import numpy  # Import numpy here, after setup_requires is processed
    metadata["include_dirs"] = [numpy.get_include()]  # Set include_dirs dynamically
    check_package_status("numpy", NUMPY_MIN_VERSION)
    setuptools.setup(**metadata)

if __name__ == "__main__":
    try:
        setup_package()
    except ext_errors as ext:
        log.warning(f"{ext}")
        log.warning("Failed installation.")
