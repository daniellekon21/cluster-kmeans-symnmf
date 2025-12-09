from setuptools import setup, Extension

symnmf_module = Extension(
    'mysymnmf',  
    sources=['symnmfmodule.c', 'symnmf.c']
)

setup(
    name='mysymnmf',
    version='1.0',
    description='Symmetric Nonnegative Matrix Factorization C Extension',
    ext_modules=[symnmf_module]
)