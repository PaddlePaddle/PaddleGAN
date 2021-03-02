# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
from setuptools import find_packages
from io import open

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()


def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='ppgan',
    packages=find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["paddlegan= paddlegan.paddlegan:main"]},
    author='PaddlePaddle Author',
    version='2.0.0-beta',
    install_requires=requirements,
    license='Apache License 2.0',
    description='Awesome GAN toolkits based on PaddlePaddle',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/PaddlePaddle/PaddleGAN',
    download_url='https://github.com/PaddlePaddle/PaddleGAN.git',
    keywords=['gan paddlegan'],
    classifiers=[
        'Intended Audience :: Developers', 'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7', 'Topic :: Utilities'
    ],
)
