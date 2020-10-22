#!/usr/bin/python3

import os
import sys

class CatchOutErr:
    def __init__(self):
        self.value = ''
    def write(self, txt):
        self.value += txt

oldstdout = sys.stdout
oldstderr = sys.stderr
catchOutErr = CatchOutErr()

# def init():
#     oldstdout = sys.stdout
#     oldstderr = sys.stderr
#     catchOutErr = CatchOutErr()

def start():
    sys.stdout = catchOutErr
    sys.stderr = catchOutErr

def end():
    sys.stdout = oldstdout
    sys.stderr = oldstderr

def main():
    # init()
    start()

    print("HelloWorld")

    end()

    print(catchOutErr.value)


if __name__ == "__main__":
    main()