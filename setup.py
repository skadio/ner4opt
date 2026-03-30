import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join("ner4opt", "_version.py")) as fp:
    exec(fp.read())

install_requires = [
    "inflect>=5.0.0",
    "nltk>=3.7",
    "overrides>=6.1.0",
    "pandas>=1.3.0",
    "simpletransformers>=0.63.7",
    "tokenizers>=0.12.1",
    "spacy>=3.2.0",
    "spacy-transformers>=1.1.7",
    "sklearn_crfsuite>=0.3.6",
    "torch>=1.12.0",
    "typing_extensions>=4.4.0",
]

setuptools.setup(
    name="ner4opt",
    description="NER4OPT Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    url="https://github.com/skadio/ner4opt",
    packages=(["models"] + setuptools.find_packages()),
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.8",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Source": "https://github.com/skadio/ner4opt",
    },
)
