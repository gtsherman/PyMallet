import os
import sys

_version = sys.version_info[0]

try:
    from setuptools import Extension, setup
except ImportError:
    sys.exit('You must have setuptools to install pymallet')

USE_CYTHON = True
try:
    from Cython.Build import cythonize
except:
    USE_CYTHON = False
    
from distutils.command.build import build as build_orig

ext = '.pyx' if USE_CYTHON else '.c'
EXTENSIONS = [Extension(name='pymallet.topicmodel', sources=['pymallet/topicmodel{}'.format(ext)])]
if USE_CYTHON:
    print('Using Cython')
    EXTENSIONS = cythonize(EXTENSIONS)

class build(build_orig):

    def finalize_options(self):
        super().finalize_options()
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        EXTENSIONS[0].include_dirs.append(numpy.get_include())

HERE = os.path.dirname(os.path.realpath(__file__))

README = open(os.path.join(HERE, 'README.md')).read()

setup(
    name='dlatk-pymallet',
    version='1.0.0',
    description='A PyMallet implementation adapted to DLATK.',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/gtsherman/pymallet',
    author='David Mimno (original PyMallet), Garrick Sherman (adaptation)',
    author_email='garricks@sas.upenn.edu',
    license='MIT',
    setup_requires=['numpy'],
    install_requires=['numpy'],
    packages=['pymallet'],
    entry_points={'console_scripts': ['pymallet=pymallet.lda:main']},
    ext_modules=EXTENSIONS,
    cmdclass={'build': build},
    include_package_data=True
)
