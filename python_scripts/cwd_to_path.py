#!/usr/bin/python3

import sys, os

def appendCWDtoSYSPATH():
    sys.path.append(os.getcwd())

def main():
    print(sys.path)
    print(os.getcwd())

    #Write Code to append CWD to Path if not already there. 

if __name__ == "__main__":
    main()