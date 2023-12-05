from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
     name='OpenMerlin',  
     version='0.1',
     author="Jimmy Li, Dmitriy Rivkin, Amal Feriani, Saria Al Lahham",
     author_email="jimmyli@cim.mcgill.ca, d.rivkin@samsung.com, amal.feriani@samsung.com, s.allahham@samsung.com",
     description="Open source Merlin project",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/SAIC-MONTREAL/MERLIN",
     package_dir={"": "src/py/"}, 
     packages=['saic5g'],
     install_requires=[
        'numpy==1.18.5',
        'pandas==1.1.4',
        'gym==0.17.3',
        'requests==2.24.0',
        'hydra-core==1.2.0'
     ],
     python_requires=">=3.5, <3.9",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )