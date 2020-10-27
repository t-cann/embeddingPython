import os
import sys

def greeting(name):
    print("Hello, " + name)

def array_tutorial(a):
    print("a.shape={}, a.dtype={}".format(a.shape, a.dtype))
    print(a)
    a *= 2
    return a[-1]

def appendCWDtoSYSPATH():
    sys.path.append(os.getcwd())