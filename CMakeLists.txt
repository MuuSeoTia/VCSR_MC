cmake_minimum_required(VERSION 3.21)
project(spmm_vcsr LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

# gfx90a = MI210/MI250, gfx942 = MI300. Matrix cores (MFMA) need gfx908 or newer.
if(NOT DEFINED GPU_ARCH)
  set(GPU_ARCH gfx90a)
endif()
set(CMAKE_HIP_ARCHITECTURES ${GPU_ARCH})

set(SPMM_SOURCES
  src/main.cpp
  src/kernels.cu
  src/vcsr_mc.cu
)

set_source_files_properties(src/kernels.cu src/vcsr_mc.cu PROPERTIES LANGUAGE HIP)

add_executable(spmm_vcsr ${SPMM_SOURCES})
target_include_directories(spmm_vcsr PRIVATE include)

target_compile_options(spmm_vcsr PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
  $<$<COMPILE_LANGUAGE:HIP>:-O3;-ffast-math>
)

find_package(hip REQUIRED)
target_link_libraries(spmm_vcsr PRIVATE hip::device)


option(USE_ROCSPARSE "Build the hipSPARSE vendor baseline" ON)
if(USE_ROCSPARSE)
  find_package(hipsparse QUIET)
  if(hipsparse_FOUND)
    target_compile_definitions(spmm_vcsr PRIVATE HAVE_ROCSPARSE=1)
    target_link_libraries(spmm_vcsr PRIVATE roc::hipsparse)
  else()
    message(WARNING "hipSPARSE not found -- building without a vendor baseline")
  endif()
endif()
