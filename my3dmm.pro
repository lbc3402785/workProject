QT -= gui

CONFIG += c++14 console
CONFIG -= app_bundle
OBJECTS_DIR=$${PWD}/build
TORCH_LIBRARY_DIRS=D:\soft\libtorch\lib
TORCH_INCLUDE_DIRS=D:\soft\libtorch\include
CUDA_INCLUDE_DIRS="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include"
CUDA_LIBRARY_DIRS="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64"
BOOST_LIBRARY_DIRS=D:\soft\boost_1_69_0\stage-static\lib
OPENCV_LIBRARY_DIRS=D:\soft\opencv3\build\x64\vc14\lib
FLANN_LIBRARY_DIRS=D:\soft\flann-1.8.4-src\lib
FLANN_INCLUDE_DIRS=D:\soft\flann-1.8.4-src\src\cpp
#FLANN_LIBRARY_DIRS=D:\soft\flann\build-vs2015\lib\Release
#FLANN_INCLUDE_DIRS=D:\soft\flann\src\cpp
# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.

DEFINES += QT_DEPRECATED_WARNINGS
INCLUDEPATH+=D:\soft\boost_1_69_0 \
                D:\soft\opencv3\build\include \
                $$(DLIB_ROOT)\include \
                D:\soft\eigen-eigen-323c052e1731 \
                D:\soft\OpenMesh8.0\include \
                D:\soft\zlib\include \
                $${PWD}\src
INCLUDEPATH+=$$FLANN_INCLUDE_DIRS
INCLUDEPATH+=$$CUDA_INCLUDE_DIRS
INCLUDEPATH+=$$TORCH_INCLUDE_DIRS
INCLUDEPATH+=$$TORCH_INCLUDE_DIRS\torch\csrc\api\include
INCLUDEPATH+=D:\soft\glm
message($$INCLUDEPATH)

QMAKE_CFLAGS_DEBUG += -MD

QMAKE_CXXFLAGS_DEBUG += -MD

LIBS+= -L$$BOOST_LIBRARY_DIRS -llibboost_filesystem-vc141-mt-x64-1_69
LIBS+=-L$$OPENCV_LIBRARY_DIRS -lopencv_world344
LIBS+= -L$$BOOST_LIBRARY_DIRS -llibboost_system-vc141-mt-x64-1_69
LIBS+=-L$$BOOST_LIBRARY_DIRS -llibboost_filesystem-vc141-mt-x64-1_69
LIBS+=-L$$BOOST_LIBRARY_DIRS -llibboost_program_options-vc141-mt-x64-1_69
LIBS+=-L$$(DLIB_ROOT)\lib -ldlib19.16.99_release_64bit_msvc1900
LIBS+=-L$$FLANN_LIBRARY_DIRS -lflann
LIBS+= -l$$TORCH_LIBRARY_DIRS\torch \
    -l$$TORCH_LIBRARY_DIRS\c10_cuda \
    -l$$TORCH_LIBRARY_DIRS\c10 \
    -l$$CUDA_LIBRARY_DIRS\cudart \
    -l$$CUDA_LIBRARY_DIRS\cudart \
    -l$$CUDA_LIBRARY_DIRS\cublas \
    -l$$CUDA_LIBRARY_DIRS\cufft \
    -l$$CUDA_LIBRARY_DIRS\curand \
    -l$$CUDA_LIBRARY_DIRS\cudnn

LIBS+=-l$$TORCH_LIBRARY_DIRS\caffe2_gpu
LIBS+=-l$$TORCH_LIBRARY_DIRS\caffe2
# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0
message($$LIBS)
SOURCES += \
        common/knnsearch.cpp \
        common/torchfunctions.cpp \
        facemorph.cpp \
        glmfunctions.cpp \
        main.cpp \
        multifitting.cpp \
        src/Dlib.cpp \
        src/cnpy.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    common/dataconvertor.h \
    common/eigenfunctions.h \
    common/knnsearch.h \
    common/raytriangleintersect.h \
    common/torchfunctions.h \
    contour.h \
    facemorph.h \
    glmfunctions.h \
    multifitting.h \
    src/Dlib.h \
    src/EigenUtil.h \
    src/NumpyUtil.h \
    src/Poisson.h \
    src/cnpy.h \
    src/mmtensorsolver.h \
    src/stdafx.h
