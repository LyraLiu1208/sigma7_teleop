QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle
#CONFIG -= x86_64 ppc64
#CONFIG += x86 ppc

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


#INCLUDEPATH += /Users/luka/Documents/work/TUD/sigma.7/codes/sigma7interface
INCLUDEPATH += "../../sdk-3.7.3/include/"
#INCLUDEPATH += "../../sdk-3.7.3/external/Eigen/Eigen/"
#INCLUDEPATH += "../../sdk-3.7.3/external/Eigen/lapack/"
#INCLUDEPATH += "../../sdk-3.7.3/external/Eigen/blas/"

#INCLUDEPATH += "../../sdk-3.7.3//build/"



SOURCES += main.cpp

LIBS += /home/luka/Documents/sigma7/sdk-3.7.3/lib/libdhd.a

#QMAKE_LFLAGS += -F/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/System/Library/Frameworks/

#LIBS += -framework CoreFoundation
#LIBS += -framework IOKit
LIBS += -lusb-1.0


# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
