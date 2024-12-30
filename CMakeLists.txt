cmake_minimum_required(VERSION 3.12.0)

project(DDStore)
# set (CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)

# find_package(Boost COMPONENTS log log_setup)
# message(STATUS "Boost_INCLUDE_DIRS: " "${Boost_INCLUDE_DIRS}")
# message(STATUS "Boost_LIBRARIES: " "${Boost_LIBRARIES}")
# include_directories(${Boost_INCLUDE_DIRS})

pkg_check_modules(LIBFABRIC libfabric REQUIRED)
include_directories(${LIBFABRIC_INCLUDE_DIRS})
link_directories(${LIBFABRIC_LIBRARY_DIRS})

add_library(ddstore src/ddstore.cxx src/common.cxx)
set_property(TARGET ddstore PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ddstore PUBLIC MPI::MPI_CXX)
target_link_libraries(ddstore PUBLIC ${Boost_LIBRARIES})
target_link_libraries(ddstore PUBLIC ${LIBFABRIC_LIBRARIES})
target_include_directories(ddstore PUBLIC include)

add_executable (demo test/demo.cxx)
target_link_libraries (demo PUBLIC ddstore)
target_include_directories(demo PUBLIC ddstore)

# option(BUILD_PYTHON_BINDINGS "Build Python bindings using cython" ON)
# find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
# add_custom_command(
#     OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/pyddstore.cpp
#     COMMAND ${Python3_EXECUTABLE} -m cython -3 --cplus ${CMAKE_CURRENT_SOURCE_DIR}/src/pyddstore.pyx
#     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src
#     DEPENDS
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/pyddstore.pyx
# )
# add_library(pyddstore SHARED
#     ${CMAKE_CURRENT_SOURCE_DIR}/src/pyddstore.cpp
# )
# set_target_properties(pyddstore PROPERTIES PREFIX "")
# set_target_properties(pyddstore PROPERTIES POSITION_INDEPENDENT_CODE ON)
# target_link_libraries(pyddstore PUBLIC Python3::Python Python3::NumPy)
# target_link_libraries(pyddstore PUBLIC MPI::MPI_CXX)
# target_link_libraries(pyddstore PUBLIC ddstore)
# target_include_directories(pyddstore PUBLIC ddstore)
