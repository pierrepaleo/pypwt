#!/usr/bin/env python

# ------------------------------------------------------------------------------
# --------------------------- setup.py -----------------------------------------
# Copyright (c) 2014, Robert T. McGibbon and the Authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ------------------------------------------------------------------------------

import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy

def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
                    'home':home, 'nvcc':nvcc,
                    'include': pjoin(home, 'include'),
                    #~ 'lib64': "/usr/lib/x86_64-linux-gnu/"
                  }

    #~ cudaconfig = {'home':home, 'nvcc':nvcc,
                  #~ 'include': pjoin(home, 'include'),
                  #~ 'lib64': pjoin(home, 'lib64')}


    for k, v in cudaconfig.items():#iteritems():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))
    return cudaconfig

CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

ext = Extension('pycudwt',
    sources = [
        'pdwt/src/wt.cu', 'pdwt/src/common.cu', 'pdwt/src/utils.cu',
        'pdwt/src/separable.cu', 'pdwt/src/nonseparable.cu', 'pdwt/src/haar.cu',
        'pdwt/src/filters.cpp',
        'src/pypwt.pyx'
    ],
    #~ library_dirs=[CUDA['lib64']],
    libraries=['cudart', 'cublas'],
    language='c++',
    #~ runtime_library_dirs=[CUDA['lib64']],
    # this syntax is specific to this build system
    # we're only going to use certain compiler args with nvcc and not with gcc
    # the implementation of this trick is in customize_compiler() below
    extra_compile_args={'gcc': [],
                        'nvcc': ['--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
    include_dirs = [numpy_include, CUDA['include'], 'src']
)


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu': ########
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            #~ self.set_executable('linker_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        #~ self.set_executable('linker_so', CUDA['nvcc']) # TEST

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


def customize_linker_for_nvcc(self):
    """
    Same as customize_compiler_for_nvcc, but for linker
    """

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_linker_so = self.linker_so
    super = self.link

    # now redefine the link method.
    def _link(self, target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None):
        if 1: ########
            self.set_executable('linker_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            raise RuntimeError("")
            #~ postargs = extra_postargs['gcc']

        super(target_desc, objects,
             output_filename, output_dir=None, libraries=None,
             library_dirs=None, runtime_library_dirs=None,
             export_symbols=None, debug=0, extra_preargs=None,
             extra_postargs=None, build_temp=None, target_lang=None)
        # reset the default likner_so, which we might have changed for cuda
        self.linker_so = default_linker_so

    # inject our redefined _compile method into the class
    self.link = _link


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


setup(
    name='pycudwt',
    author='Pierre Paleo',
    version='1.0.0',
    author_email = "pierre.paleo@esrf.fr",
    maintainer = "Pierre Paleo",
    maintainer_email = "pierre.paleo@esrf.fr",

    install_requires = [
        'Cython',
        'numpy',
    ],

    long_description = """
    Python Wrapper for Cuda Discrete Wavelet Transform
    """,

    ext_modules = [ext],
    # inject our custom trigger
    cmdclass={'build_ext': custom_build_ext},

    # since the package has c code, the egg cannot be zipped
    zip_safe=False
)
