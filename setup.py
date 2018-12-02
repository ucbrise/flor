import setuptools
import io
with io.open("README.md", mode="r", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as f:
    requirements = f.read().split('\n')

setuptools.setup(
     name='pyflor',
     version='0.0.1-alpha',
     author="Rolando Garcia",
     author_email="rogarcia@berkeley.edu",
     scripts=['pyflor'],
     description="A context-centric logger and automatic version controller",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ucbrise/flor",
     packages=setuptools.find_packages(),
     install_requires = requirements,
     classifiers = [
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
 )
