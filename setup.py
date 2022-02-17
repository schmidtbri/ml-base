from os.path import abspath, dirname, join
from io import open
from setuptools import setup, find_packages

# Get the long description from the README file
here = abspath(dirname(__file__))

with open(join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(join(here, "ml_base", "version.txt"), encoding="utf-8") as f:
    version = f.read()

setup(name="ml_base",
      version=version,
      author="Brian Schmidt",
      author_email="6666331+schmidtbri@users.noreply.github.com",
      description="Base classes and utilities that are useful for deploying ML models.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/schmidtbri/ml-base",
      packages=find_packages(exclude=["tests", "*tests", "tests*"]),
      python_requires=">=3.5",
      install_requires=["pydantic>=1.5"],
      tests_require=["pytest", "pytest-html", "pylama", "coverage", "coverage-badge", "bandit", "safety", "pytype"],
      package_data={
          "ml_base": [
                  "version.txt"
            ]
      },
      project_urls={
          "Documentation": "https://schmidtbri.github.io/ml-base/",
          "Code": "https://github.com/schmidtbri/ml-base",
          "Tracker": "https://github.com/schmidtbri/ml-base/issues"
      },
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ])
