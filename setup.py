import setuptools
import numpy

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="slearn",
    packages=["slearn"],
    version="0.0.9",
    setup_requires=["numpy>=1.20.0"],
    install_requires=["numpy>=1.20.0", "scikit-learn", "pandas", "fABBA", "lightgbm", "requests"],
    include_dirs=[numpy.get_include()],
    long_description=long_description,
    author="Stefan Güttel, Xinye Chen, Roberto Cahuantzi",
    author_email="stefan.guettel@manchester.ac.uk, xinye.chen@manchester.ac.uk, Roberto.Cahuantzi@manchester.ac.uk",
    classifiers=["Intended Audience :: Science/Research",
                "Intended Audience :: Developers",
                "Programming Language :: Python",
                "Topic :: Software Development",
                "Topic :: Scientific/Engineering",
                "Operating System :: Microsoft :: Windows",
                "Operating System :: Unix",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                ],
    maintainer="Xinye Chen",
    maintainer_email="xinye.chen@manchester.ac.uk",
    description="A package linking symbolic representation with sklearn for time series prediction",
    long_description_content_type='text/markdown',
    url="https://github.com/nla-group/slearn.git",
    license='MIT License'
)
