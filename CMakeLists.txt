cmake_minimum_required(VERSION 3.1)
#SET(CMAKE_TOOLCHAIN_FILE "D:/soft/vcpkg/scripts/buildsystems/vcpkg.cmake")
project(My3DMM CXX)

add_definitions(-std=c++14)
set(CMAKE_CXX_STANDARD 14)

# Set OpenCVConfig.cmake Search Directory
#set(BOOST_ROOT "F:/vcpkg/installed/x64-windows") 
#set(OpenCV_DIR "F:/vcpkg/installed/x64-windows/share/opencv" )
#set(VCPKG_TARGET_TRIPLET "x64-windows")


set(CMAKE_CXX_FLAGS_DEBUG "-g3 -O0 -pg")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -DNDEBUG -march=native")


SET(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#if(NOT Torch_FOUND)
#    message(FATAL_ERROR "Pytorch Not Found!")
#endif(NOT Torch_FOUND)
find_package(PythonLibs 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system python37 numpy)
find_package(Eigen3 REQUIRED )
find_package(dlib REQUIRED)
FIND_PACKAGE(CUDA)
# openmesh
list (APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_module)
find_package( OpenMesh REQUIRED)



include_directories( ${OpenMesh_INCLUDE_DIRS} )



include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${dlib_INCLUDE_DIRS})

link_directories(${Boost_LIBRARY_DIRS})
link_directories(${PYTHON_LIBRARIES})

#set(Boost_USE_STATIC_LIBS OFF) 
#set(Boost_USE_MULTITHREADED ON)  
#set(Boost_USE_STATIC_RUNTIME OFF) 

file(GLOB ALLCPP
    "src/*.cpp"
)
file(GLOB COMMON
    "common/*.cpp"
)

#add_executable(TestProject ${ALLCPP} ${COMMON})
#target_link_libraries(TestProject Eigen3::Eigen ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} dlib::dlib)

add_executable(MultiFace main.cpp)
target_link_libraries(MultiFace ${OpenMesh_LIBS} OpenMeshCore OpenMeshTools) 
#target_link_libraries(SimpleFace Eigen3::Eigen ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} dlib::dlib_shared)
target_link_libraries(MultiFace Eigen3::Eigen ${CUDA_LIBS} ${OpenCV_LIBS} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${TORCH_LIBRARIES} dlib::dlib)


