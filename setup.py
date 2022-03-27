import setuptools
import numpy

with open("README.rst", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="slearn",
    packages=["slearn"],
    version="0.2.4",
    setup_requires=["numpy"],
    install_requires=["numpy", "scikit-learn", "pandas", "lightgbm", "requests"],
    include_dirs=[numpy.get_include()],
    long_description=long_description,
    author="nla-group",
    maintainer="nla-group",
    author_email="stefan.guettel@manchester.ac.uk",
    classifiers=["Intended Audience :: Science/Research",
                "Intended Audience :: Developers",
                "Programming Language :: Python",
                "Topic :: Software Development",
                "Topic :: Scientific/Engineering",
                "Operating System :: Microsoft :: Windows",
                "Operating System :: Unix",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: 3.6",
                "Programming Language :: Python :: 3.7",
                "Programming Language :: Python :: 3.8",
                "Programming Language :: Python :: 3.9",
                ],
    maintainer_email="stefan.guettel@manchester.ac.uk",
    description="A package linking symbolic representation with sklearn for time series prediction",
    long_description_content_type='text/x-rst',
    url="https://github.com/nla-group/slearn.git",
    license='MIT License'
)
