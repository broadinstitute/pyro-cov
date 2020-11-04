import sys

from setuptools import find_packages, setup

try:
    long_description = open('README.md', encoding='utf-8').read()
except Exception as e:
    sys.stderr.write('Failed to read README.md: {}\n'.format(e))
    sys.stderr.flush()
    long_description = ''

setup(
    name='pyrophylo',
    version='0.0.0',
    description='Pyro tools for phylogenetic inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(include=['pyrophylo']),
    url='http://pyro.ai',
    author='Pyro team at the Broad Institute of MIT and Harvard',
    author_email='fobermey@broadinstitute.org',
    install_requires=[
        'biopython>=1.54',
        'pyro-ppl>=1.5',
        'geopy',
        'gpytorch',
        'scikit-learn',
    ],
    extras_require={
        'test': [
            'flake8',
            'pytest>=5.0',
        ],
    },
    python_requires='>=3.6',
    keywords='pyro pytorch phylogenetic machine learning',
    license='Apache 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
