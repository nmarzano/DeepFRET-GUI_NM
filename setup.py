from setuptools import setup, find_packages
setup(
    name='DeepFRET_GUI_NM',
    version='1.0.0',
    url='https://github.com/nmarzano/DeepFRET-GUI_NM',
    author='Nicholas Marzano',
    author_email='nmarzano@uow.edu.au',
    description='Package to be used for fitting FRET data with HMM model',
    packages=find_packages(),    
    # install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
)