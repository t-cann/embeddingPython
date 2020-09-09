# Embeddign Python in C++ Example

Changes to default C/C++ Configuration 
- Include Path of Python.h Header file "/usr/include/python3.8/**"


> python3.8-config --cflags

'-I/usr/include/python3.8 -I/usr/include/python3.8  -Wno-unused-result -Wsign-compare -g -fdebug-prefix-map=/build/python3.8-fKk4GY/python3.8-3.8.2=. -specs=/usr/share/dpkg/no-pie-compile.specs -fstack-protector -Wformat -Werror=format-security  -DNDEBUG -g -fwrapv -O3 -Wall'

> python3.8-config --cflags

'-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu -L/usr/lib  -lcrypt -lpthread -ldl  -lutil -lm -lm' 

