cmake_minimum_required(VERSION 3.10)

if (OPENNI2_FOUND)
  include_directories(${OPENNI2_INCLUDE_PATHS})
  add_executable(oni2raw oni2raw.cpp)
  target_link_libraries(oni2raw ${OPENNI2_LIBRARIES})
endif(OPENNI2_FOUND)

add_library(lodepng SHARED lodepng.cpp)
target_include_directories(lodepng PUBLIC .)

add_executable(scene2raw scene2raw.cpp)
target_link_libraries(scene2raw lodepng)
target_include_directories(scene2raw PUBLIC .)

add_executable(tum2raw tum2raw.cpp)
target_link_libraries(tum2raw lodepng stdc++fs)
target_include_directories(tum2raw PUBLIC .)
