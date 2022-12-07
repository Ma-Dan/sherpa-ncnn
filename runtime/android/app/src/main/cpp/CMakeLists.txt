cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(sherpa)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/jniLibs/${ANDROID_ABI})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/kaldi-native-fbank)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/ncnn)

include(kaldi-native-fbank)
include(ncnn)
ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/sherpa-ncnn)
