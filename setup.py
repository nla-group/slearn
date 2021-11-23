import setuptools
import numpy

with open("README.md", 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="slearn",
    packages=["slearn"],
    version="0.0.1",
    setup_requires=["numpy>=1.20.0"],
    install_requires=["numpy>=1.20.0", "scikit-learn", "pandas", "fABBA", "lightgbm"],
    include_dirs=[numpy.get_include()],
    long_description=long_description,
    author="Stefan Guettel, Xinye Chen",
    author_email="stefan.guettel@manchester.ac.uk",
    description="A package linking symbolic representation with sklearn for time series prediction",
    long_description_content_type='text/markdown',
    url="https://github.com/nla-group/slearn.git",
    license='MIT License'
)
