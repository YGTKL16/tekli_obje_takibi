# Install script for directory: /home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/ykula/tracker/third_party/eigen-install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/AdolcForward"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/AlignedVector3"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/ArpackSupport"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/AutoDiff"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/BVH"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/EulerAngles"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/FFT"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/IterativeSolvers"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/KroneckerProduct"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/LevenbergMarquardt"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/MatrixFunctions"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/MoreVectorization"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/MPRealSupport"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/NonLinearOptimization"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/NumericalDiff"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/OpenGLSupport"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/Polynomials"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/Skyline"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/SparseExtra"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/SpecialFunctions"
    "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/Splines"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Devel" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "/home/ykula/tracker/third_party/eigen-3.4.0/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/ykula/tracker/third_party/eigen-build/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "/home/ykula/tracker/third_party/eigen-build/unsupported/Eigen/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
