#!/usr/bin/env bash
set -ex

dir=build-ios

mkdir -p $dir
cd $dir

cmake -DCMAKE_TOOLCHAIN_FILE="toolchains/ios.toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DSHERPA_NCNN_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_NCNN_ENABLE_BINARY=OFF \
    -DPLATFORM=OS64 \
    -DIOS=ON \
    -DIOS_PLATFORM=OS64 \
    -DIOS_ARCH="arm64" \
    -DENABLE_BITCODE=0 \
    -DCMAKE_INSTALL_PREFIX=./install ..
make VERBOSE=1 -j4
make install/strip
