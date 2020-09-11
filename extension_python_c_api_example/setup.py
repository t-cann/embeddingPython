#!/usr/bin/env python3 
from distutils.core import Extension, setup

module = Extension("spam", sources=[ "spammodule.c" ])
setup(name="spam", version="1.0", ext_modules=[ module ])
