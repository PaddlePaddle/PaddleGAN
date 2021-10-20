import io
import os
from os import path
import re
from setuptools import setup, find_packages
# To use consisten encodings
from codecs import open

# Function from: https://github.com/pytorch/vision/blob/master/setup.py


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

# Function from: https://github.com/pytorch/vision/blob/master/setup.py


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = readme_file.read()

VERSION = find_version('face_alignment', '__init__.py')

requirements = [
    'torch',
    'numpy',
    'scipy>=0.17',
    'scikit-image',
    'opencv-python',
    'tqdm',
    'numba',
    'enum34;python_version<"3.4"'
]

setup(
    name='face_alignment',
    version=VERSION,

    description="Detector 2D or 3D face landmarks from Python",
    long_description=long_description,
    long_description_content_type="text/markdown",

    # Author details
    author="Adrian Bulat",
    author_email="adrian@adrianbulat.com",
    url="https://github.com/1adrianb/face-alignment",

    # Package info
    packages=find_packages(exclude=('test',)),

    python_requires='>=3',
    install_requires=requirements,
    license='BSD',
    zip_safe=True,

    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',

        # Supported python versions
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
