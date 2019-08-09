import setuptools
import io

with io.open("README.md", mode="r", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt", 'r') as f:
    requirements = f.read().split('\n')

setuptools.setup(
     name='pyflor',
     version='0.0.3a16',
     author="Rolando Garcia",
     author_email="rogarcia@berkeley.edu",
     description="A diagnostics and sharing system for ML",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/ucbrise/flor",
     packages=[p for p in setuptools.find_packages() if p != 'tests'],
     install_requires = requirements,
     classifiers = [
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: Apache Software License",
         "Operating System :: OS Independent",
     ],
    entry_points={
        'console_scripts': [
            'pyflor = flor.__main__:main',
            'pyflor_install = flor.__main__:install',
            'pyflor_uninstall = flor.__main__:uninstall'

        ]
    }
 )
