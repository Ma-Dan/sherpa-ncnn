include_directories(${CMAKE_SOURCE_DIR})
include_directories(${kaldi_native_fbank_SOURCE_DIR})
include_directories(${ncnn_SOURCE_DIR})

find_library(log-lib log)
find_library(android-lib android)
find_library(vulkan-lib vulkan)
find_library(jnigraphics-lib jnigraphics)

set(sherpa_srcs
  decode.cc
  features.cc
  lstm-model.cc
  conv-emformer-model.cc
  model.cc
  symbol-table.cc
  sherpa.cc
)
add_library(sherpa SHARED ${sherpa_srcs})
target_link_libraries(sherpa
        kaldi-native-fbank-core
        ${log-lib}
        ${vulkan-lib}
        ${android-lib}
        ${jnigraphics-lib}
        ncnn)
