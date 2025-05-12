import setuptools
import platform
import logging

PACKAGE_NAME = "slearn"
VERSION = "0.2.6"
SETUP_REQUIRES = ["numpy>=1.17.2"]
INSTALL_REQUIRES = [
    "numpy>=1.17.2",
    "scikit-learn>=1.0.0",
    "pandas>=1.0.0",
    "requests>=2.25.0"
]
MAINTAINER = "NLA Group"
EMAIL = "stefan.guettel@manchester.ac.uk"
AUTHORS = "Roberto Cahuantzi, Xinye Chen, Stefan Güttel"

# Read README.rst
try:
    with open("README.rst", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A package linking symbolic representation with sklearn for time series prediction"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set numpy minimum version based on Python implementation
NUMPY_MIN_VERSION = "1.19.2" if platform.python_implementation() == "PyPy" else "1.17.2"

# Package metadata
metadata = {
    "name": PACKAGE_NAME,
    "version": VERSION,
    "packages": setuptools.find_packages(),
    "setup_requires": SETUP_REQUIRES,
    "install_requires": INSTALL_REQUIRES,
    "include_package_data": True,
    "long_description": long_description,
    "long_description_content_type": "text/x-rst",
    "author": AUTHORS,
    "maintainer": MAINTAINER,
    "author_email": EMAIL,
    "maintainer_email": EMAIL,
    "description": "A package linking symbolic representation with sklearn for time series prediction",
    "url": "https://github.com/nla-group/slearn",
    "license": "MIT",
    "classifiers": [
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "License :: OSI Approved :: MIT License",
    ],
    "python_requires": ">=3.7",
}

def setup_package():
    """Configure and run the package setup."""
    try:
        # Removed numpy check and include_dirs to avoid import during sdist
        setuptools.setup(**metadata)
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        raise

if __name__ == "__main__":
    setup_package()