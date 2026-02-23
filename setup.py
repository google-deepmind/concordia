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
    version='2.3.1',
    license='Apache 2.0',
    license_files=['LICENSE'],
    url='https://github.com/google-deepmind/concordia',
    download_url='https://github.com/google-deepmind/concordia/releases',
    author='DeepMind',
    author_email='noreply@google.com',
    description=(
        'A library for building a generative model of social interacions.'
    ),
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
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    packages=setuptools.find_packages(include=['concordia', 'concordia.*']),
    package_data={},
    python_requires='>=3.12',
    install_requires=(
        'absl-py',
        'ipython',
        'matplotlib',
        'numpy>=1.26',
        'pandas',
        'python-dateutil',
        'reactivex',
        'tenacity',
        'termcolor',
    ),
    extras_require={
        'amazon': [
            # Used in contrib.language_models.amazon
            'boto3',
        ],
        'dev': [
            # Used in development
            'build',
            'isort',
            'jupyter',
            'pip-tools',
            'pyink',
            'pylint',
            'pytest-xdist',
            'pytype',
            'twine',
        ],
        'google': [
            # Used in concordia.contrib.language_models.google
            'google-genai',
            'google-cloud-aiplatform',  # For custom endpoint models
        ],
        'huggingface': [
            # Used in concordia.contrib.language_models.huggingface
            'accelerate',
            'torch',
            'transformers',
        ],
        'langchain': [
            # Used in concordia.contrib.language_models.langchain
            'langchain-community',
        ],
        'mistralai': [
            # Used in concordia.contrib.language_models.mistralai
            'mistralai',
        ],
        'ollama': [
            # Used in concordia.contrib.language_models.ollama
            'ollama',
        ],
        'openai': [
            # Used in concordia.contrib.language_models.openai
            'openai>=1.3.0',
        ],
        'together': [
            # Used in concordia.contrib.language_models.together
            'together',
        ],
        'vllm': [
            # Used in concordia.contrib.language_models.vllm
            'vllm',
        ],
        'groq': [
            # Used in concordia.contrib.language_models.groq
            'groq',
        ],
    },
)
