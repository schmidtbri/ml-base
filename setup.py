from os.path import abspath, dirname, join
from io import open
from setuptools import setup, find_packages

# Get the long description from the README file
here = abspath(dirname(__file__))


def load_file(file_name):
    here = abspath(dirname(__file__))
    with open(join(here, file_name)) as f:
        return f.read()


setup(name="ml_base",
      version=load_file("ml_base/version.txt"),
      author="Brian Schmidt",
      author_email="6666331+schmidtbri@users.noreply.github.com",
      description="Base classes and utilities that are useful for deploying ML models.",
      long_description=load_file("README.md"),
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
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ],
      keywords=[
          "machine learning", "REST", "service", "model deployment"
      ])
