from os import path
from io import open
from setuptools import setup, find_packages

from ml_base import __name__, __version__, __doc__

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name=__name__,
      version=__version__,
      author="Brian Schmidt",
      author_email="6666331+schmidtbri@users.noreply.github.com",
      description=__doc__,
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/schmidtbri/ml-base",
      packages=find_packages(exclude=["tests"]),
      python_requires=">=3.5",
      install_requires=["pydantic>=1.5"],
      tests_require=['pytest', 'pytest-html', 'pylama', 'coverage', 'coverage-badge', 'bandit', 'safety', "pytype"],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ])
