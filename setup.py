import setuptools
import io
with io.open("README.md", mode="r", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as f:
    requirements = f.read().split('\n')

setuptools.setup(
     name='flor',
     version='0.0.0-alpha',
     author="Rolando Garcia",
     author_email="rogarcia@berkeley.edu",
     description="A Context Management System that feels like a logger",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ucbrise/flor",
     packages=setuptools.find_packages(),
     install_reqs = requirements,
     classifiers = [
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
 )
