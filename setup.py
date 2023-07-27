import setuptools
import io

with io.open("README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="flordb",
    version="2.7.1",
    author="Rolando Garcia",
    author_email="rogarcia@berkeley.edu",
    description="Fast Low-Overhead Recovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ucbrise/flor",
    packages=setuptools.find_packages(),
    install_requires=[
        "GitPython",
        "cloudpickle",
        "astunparse",
        "pandas",
        "bidict==0.21.3",
        "apted",
        "matplotlib",
        "scikit-learn",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
