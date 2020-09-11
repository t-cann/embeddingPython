#!/usr/bin/env python3 
from distutils.core import Extension, setup

module = Extension("test_module", sources=[ "module_source.c" ])
setup(name="test_module", version="1.0", ext_modules=[ module ])
