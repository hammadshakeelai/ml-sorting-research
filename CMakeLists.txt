cmake_minimum_required(VERSION 3.20)
project(LearnedSortML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W4)
else()
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

add_executable(learned_sort learned_sort.cpp)
