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

def main():
    print("Hello World!")

def setup():
    
    sys.stdout = catchOutErr
    sys.stderr = catchOutErr

def end():
    sys.stdout = oldstdout
    sys.stderr = oldstderr


def appendCWDtoSYSPATH():
    sys.path.append(os.getcwd())

def multiply(a,b):
    print("Will compute", a, "times", b)
    c = 0
    for i in range(0, a):
        c = c + b
    return c

if __name__ == "__main__":
    main()