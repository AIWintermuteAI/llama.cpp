set(SIMPLE_APP llama-vision)
add_executable(${SIMPLE_APP} vision.cpp)
install(TARGETS ${SIMPLE_APP} RUNTIME)
target_link_libraries(${SIMPLE_APP} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${SIMPLE_APP} PRIVATE cxx_std_17)

set(SOCKET_APP llama-vision-socket)
add_executable(${SOCKET_APP} vision_socket.cpp)
install(TARGETS ${SOCKET_APP} RUNTIME)
target_link_libraries(${SOCKET_APP} PRIVATE llama common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${SOCKET_APP} PRIVATE cxx_std_17)

