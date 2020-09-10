def filterFunc(ln):
    if ln == "abc":
        raise Exception("xxx!")
    return ln.upper()
