# Writing C Extensions for Python 

Source https://ishantheperson.github.io/posts/python-c-ext/

```bash
python3 ./setup.py build
```

```Python
>>> import test_module
Initialization
>>> test_module
<module 'test_module' from '/home/thomas/Documents/projects/embeddingPython/WritingCExtensionsforPython/build/lib.linux-x86_64-3.8/test_module.cpython-38-x86_64-linux-gnu.so'>
>>> test_module.hello
<built-in function hello>
>>> test_module.hello
```
