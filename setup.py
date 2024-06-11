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


def _remove_excluded(description: str) -> str:
  description, *sections = description.split('<!-- GITHUB -->')
  for section in sections:
    excluded, included = section.split('<!-- /GITHUB -->')
    del excluded
    description += included
  return description


with open('README.md') as f:
  LONG_DESCRIPTION = _remove_excluded(f.read())


setuptools.setup(
    name='gdm-concordia',
    version='1.6.0',
    license='Apache 2.0',
    license_files=['LICENSE'],
    url='https://github.com/google-deepmind/concordia',
    download_url='https://github.com/google-deepmind/concordia/releases',
    author='DeepMind',
    author_email='noreply@google.com',
    description=(
        'A library for building a generative model of social interacions.'
    ),
    description_content_type='text/plain',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
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
    packages=setuptools.find_packages(include=['concordia', 'concordia.*']),
    package_data={},
    python_requires='>=3.11',
    install_requires=(
        # TODO: b/312199199 - remove some requirements.
        'absl-py',
        'google-cloud-aiplatform',
        'google-generativeai',
        'ipython',
        'langchain',
        'matplotlib',
        'mistralai',
        'numpy',
        'openai>=1.3.0',
        'pandas<=2.0.3',
        'python-dateutil',
        'reactivex',
        'retry',
        'scipy',
        'termcolor',
        'transformers',
        'typing-extensions',
    ),
    extras_require={
        # Used in development.
        'dev': [
            'build',
            'isort',
            'jupyter',
            'pipreqs',
            'pip-tools',
            'pyink',
            'pylint',
            'pytest-xdist',
            'pytype',
            'twine',
        ],
    },
)
