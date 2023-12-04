# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Install script for setuptools."""

import setuptools

setuptools.setup(
    name='gdm-concordia',
    version='1.0.0.dev.0',
    license='Apache 2.0',
    license_files=['LICENSE'],
    url='https://github.com/google-deepmind/concordia',
    download_url='https://github.com/google-deepmind/concordia/releases',
    author='DeepMind',
    author_email='noreply@google.com',
    description=(
        'A library for building a generative model of social interacions.'
    ),
    keywords=(
        'multi-agent agent-based-simulation generative-agents python'
        ' machine-learning'
    ),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    package_dir={
        'concordia': 'concordia',
    },
    package_data={},
    python_requires='>=3.11',
    install_requires=[
        # TODO: b/312199199 - remove some requirements.
        'absl-py',
        'google-cloud-aiplatform',
        'ipython~=3.2.3',
        'matplotlib~=3.6.1',
        'numpy~=1.26.2',
        'pandas>=1.5.3,<2.2.0',
        'python-dateutil~=2.8.2',
        'reactivex~=4.0.4',
        'retry~=0.9.2',
        'saxml',
        'scipy~=1.9.3',
        'tensorflow',
        'tensorflow_hub',
        'termcolor~=1.1.0',
        'typing-extensions~=4.5.0',
    ],
    extras_require={
        # Used in development.
        'dev': [
            'build',
            'isort',
            'pipreqs',
            'pyink',
            'pylint',
            'pytest-xdist',
            'pytype',
        ],
    },
)
