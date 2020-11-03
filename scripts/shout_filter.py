#!/usr/bin/python2.7

def filterFunc(ln):
    if ln == "abc":
        raise Exception("xxx!")
    return ln.upper()
