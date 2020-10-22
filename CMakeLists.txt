cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(embeddingPythonCmake LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development NumPy)

add_subdirectory(src)

add_executable(embeddingPythonCmake
    main.cpp
    mainwindow.h
    mainwindow.cpp 
    consolewidget.h 
    consolewidget.cpp 
    pyConsole.h 
    pyConsole.cpp 
    consoledialog.h 
    consoledialog.cpp
)

target_link_libraries(embeddingPythonCmake
  #Python::Python
  #Python::NumPy
  Python3::Python
  Python3::NumPy
  Qt5::Core
  Qt5::Widgets
)