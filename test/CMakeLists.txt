# include(Dart)
# include(CTest)
cmake_minimum_required(VERSION 3.8)

set(This pyConsoleTests)

set(Sources
    pyConsole_tests.cpp
)
add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC
    gtest_main
    pyConsole
)

add_test(
    NAME ${This}
    COMMAND ${This}
)

set(This gmockTests)

set(Sources
    gmock_tests.cpp
)
add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC
    gtest_main
    gmock
)

add_test(
    NAME ${This}
    COMMAND ${This}
)

set(This PythonAPITests)

set(Sources
    Python_tests.cpp
)
add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC
    gtest_main
    Python3::Python
)

add_test(
    NAME ${This}
    COMMAND ${This}
)

set(This NumPyAPITests)

set(Sources
    Numpy_tests.cpp
)
add_executable(${This} ${Sources})

target_link_libraries(${This} PUBLIC
    gtest_main
    Python3::Python
    Python3::NumPy
)

add_test(
    NAME ${This}
    COMMAND ${This}
)