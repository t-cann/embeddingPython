#!/usr/bin/python3

print("Hello, World!")

def main():
    print("Hello World!")

def multiply(a,b):
    print("Will compute", a, "times", b)
    c = 0
    for i in range(0, a):
        c = c + b
        print(i)
    return c

if __name__ == "__main__":
    main()