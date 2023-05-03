import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    required = fh.read().splitlines()

with open(os.path.join("ner4opt", "_version.py")) as fp:
    exec(fp.read())

setuptools.setup(
    name="ner4opt",
    description="NER4OPT Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    author="",
    url="",
    packages=(["models"] + setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])),
    include_package_data=True,
    install_requires=required,
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "",
    },
)
