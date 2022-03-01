from setuptools import setup, find_packages
import re

with open('README.md') as f:
    readme = f.read()

# extract version
with open('labplatform/__init__.py') as file:
    for line in file.readlines():
        m = re.match("__version__ *= *['\"](.*)['\"]", line)
        if m:
            version = m.group(1)

setup(name='slab-platform',
      version=version,
      description='High level tools for controlling hardware/running experiments',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='https://github.com/sandriver03/slab-platform',
      author='Chao Huang',
      author_email='c.huang03@gmail.com',
      license='MIT',
      python_requires='>=3.7',
      install_requires=["matplotlib < 3.5"],
      extras_require={'testing': ['pytest', 'h5netcdf'],
                      'docs': ['sphinx', 'sphinx-rtd-theme'],
                      },
      packages=find_packages(),
      # package_data={'slab': ['data/mit_kemar_normal_pinna.bz2']},
      # include_package_data=True,
      zip_safe=False)
