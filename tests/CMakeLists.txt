add_executable(tests
        main.cpp
        common.cpp
        input_size.cpp
        input_types.cpp
        stage_combinations.cpp
        workloads.cpp
        pud.cpp)
target_link_libraries(tests PRIVATE dappa)

# structs.h is embedded into the generated DPU code (see INCTXT in common.cpp).
set_source_files_properties(common.cpp PROPERTIES
        COMPILE_DEFINITIONS DAPPA_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/structs.h")

add_test(NAME dappa_tests COMMAND tests)
