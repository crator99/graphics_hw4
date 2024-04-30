cmake_minimum_required(VERSION 3.0.0)
project(graphics_4 VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include/glad
    ${CMAKE_SOURCE_DIR}/include/GLFW
    ${CMAKE_SOURCE_DIR}/include/KHR
    ${CMAKE_SOURCE_DIR}/include
)

# Include header files
set(HEADERS
    ${CMAKE_SOURCE_DIR}/include/color.h
    ${CMAKE_SOURCE_DIR}/include/ray.h
    ${CMAKE_SOURCE_DIR}/include/vec3.h
)
enable_testing()

add_executable(MyExecutable ${CMAKE_SOURCE_DIR}/src/main.cpp ${CMAKE_SOURCE_DIR}/include/GLAD/glad.c)

find_package(OpenGL REQUIRED)
target_link_libraries(MyExecutable PRIVATE OpenGL::GL)

target_link_libraries(MyExecutable PRIVATE ${CMAKE_SOURCE_DIR}/lib/libglfw3dll.a)
add_custom_command(TARGET MyExecutable POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/lib/glfw3.dll
        $<TARGET_FILE_DIR:MyExecutable>
)

set_target_properties(MyExecutable PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

set_property(TARGET MyExecutable PROPERTY CXX_STANDARD 17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)