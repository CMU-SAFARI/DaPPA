foreach(example vector_add dot_product select histogram)
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example} PRIVATE dappa)
endforeach()
