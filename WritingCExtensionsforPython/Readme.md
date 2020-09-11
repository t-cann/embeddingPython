# Writing C Extensions for Python 

Source https://ishantheperson.github.io/posts/python-c-ext/

```bash
$ python3 ./setup.py build
running build
running build_ext
building 'test_module' extension
creating build
creating build/temp.linux-x86_64-3.8
x86_64-linux-gnu-gcc -pthread -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC -I/usr/include/python3.8 -c module_source.c -o build/temp.linux-x86_64-3.8/module_source.o
creating build/lib.linux-x86_64-3.8
x86_64-linux-gnu-gcc -pthread -shared -Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 build/temp.linux-x86_64-3.8/module_source.o -o build/lib.linux-x86_64-3.8/test_module.cpython-38-x86_64-linux-gnu.so
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

What does "sid" after args in PyParseTuple mean/do
type to exspect "String , interger, double".

More Types found at https://docs.python.org/3/c-api/arg.html

https://packaging.python.org/guides/packaging-binary-extensions/ 
Interesting Discussion about 3 reasons you would make a binary extension, disadvantages and alternatives.
