cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(Embedding_Python_in_C LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()
add_subdirectory(googletest) #https://www.eriksmistad.no/getting-started-with-google-test-on-ubuntu/

# find_package(Python2 COMPONENTS Interpreter Development)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

set(Headers
  src/pyConsole.h
)
set(Sources
  src/pyConsole.cpp
)

add_library(pyConsole STATIC ${Sources} ${Headers})

target_link_libraries(pyConsole
  Python3::Python
  Python3::NumPy
)

add_subdirectory(test)
