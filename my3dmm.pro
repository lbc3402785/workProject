QT -= gui

CONFIG += console
CONFIG -= app_bundle
DEFINES += QT_DEPRECATED_WARNINGS GLOG_NO_ABBREVIATED_SEVERITIES GLM_FORCE_UNRESTRICTED_GENTYPE _HAS_CXX17=1
QMAKE_CXXFLAGS+=/openmp /std:c++17
OBJECTS_DIR=$${PWD}/build
TORCH_LIBRARY_DIRS=d:\soft\libtorch\lib
TORCH_INCLUDE_DIRS=d:\soft\libtorch\include
#CUDA_INCLUDE_DIRS="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include"
#CUDA_LIBRARY_DIRS="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64"

OPENCV_LIBRARY_DIRS=D:\soft\opencv3\build\x64\vc14\lib
#FLANN_LIBRARY_DIRS=D:\soft\flann-1.8.4-src\lib
#FLANN_INCLUDE_DIRS=D:\soft\flann-1.8.4-src\src\cpp
#FLANN_LIBRARY_DIRS=D:\soft\flann\build-vs2015\lib\Release
#FLANN_INCLUDE_DIRS=D:\soft\flann\src\cpp
# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.


INCLUDEPATH+=d:\soft\opencv3\build\include \
                $$(DLIB_ROOT)\include \
                d:\soft\eigen-eigen-323c052e1731 \
#                D:\soft\OpenMesh8.0\include \
                d:\soft\zlib\include \
                $${PWD}\src
INCLUDEPATH+=$$FLANN_INCLUDE_DIRS
#INCLUDEPATH+=$$CUDA_INCLUDE_DIRS
INCLUDEPATH+=$$TORCH_INCLUDE_DIRS
INCLUDEPATH+=$$TORCH_INCLUDE_DIRS\torch\csrc\api\include
INCLUDEPATH+=d:\soft\glm

INCLUDEPATH+="C:\Program Files (x86)\Ceres\include" \
                 "C:\Program Files (x86)\glog\include" \
                 "C:\Program Files (x86)\gflags\include"

LIBS+=-L"C:\Program Files (x86)\Ceres\lib" -lceres
LIBS+=-L"C:\Program Files (x86)\glog\lib" -lglog
LIBS+=-L"C:\Program Files (x86)\gflags\lib" -lgflags


QMAKE_CFLAGS_DEBUG += -MD

QMAKE_CXXFLAGS_DEBUG += -MD

LIBS+=-L$$OPENCV_LIBRARY_DIRS -lopencv_world344
LIBS+=-L$$(DLIB_ROOT)\lib -ldlib19.16.99_release_64bit_msvc1900
#LIBS+=-L$$FLANN_LIBRARY_DIRS -lflann
LIBS+= -l$$TORCH_LIBRARY_DIRS\torch \
    -l$$TORCH_LIBRARY_DIRS\c10_cuda \
    -l$$TORCH_LIBRARY_DIRS\c10


LIBS+=-l$$TORCH_LIBRARY_DIRS\caffe2_gpu
LIBS+=-l$$TORCH_LIBRARY_DIRS\caffe2
# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
message($$LIBS)
SOURCES += \
        common/torchfunctions.cpp \
        facemorph.cpp \
        glmfunctions.cpp \
        main.cpp \
        multifitting.cpp \
        priorcostcallback.cpp \
        src/Dlib.cpp \
        src/cnpy.cpp \
        test.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    ceresnonlinear.hpp \
    common/dataconvertor.h \
    common/eigenfunctions.h \
    common/raytriangleintersect.h \
    common/torchfunctions.h \
    contour.h \
    facemorph.h \
    glmfunctions.h \
    multifitting.h \
    priorcostcallback.h \
    src/Dlib.h \
    src/EigenUtil.h \
    src/NumpyUtil.h \
    src/Poisson.h \
    src/cnpy.h \
    src/mmtensorsolver.h \
    src/stdafx.h \
    test.h
