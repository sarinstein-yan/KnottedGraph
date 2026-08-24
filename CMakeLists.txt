cmake_minimum_required(VERSION 3.15)
project(KnottedGraphNative LANGUAGES NONE)

include(CheckLanguage)
check_language(CXX)

if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Boost REQUIRED)

    pybind11_add_module(_yamada_native MODULE src/knotted_graph/invariants/yamada/_yamada_native.cpp)
    pybind11_add_module(_yamada_frontier MODULE src/knotted_graph/invariants/yamada/_yamada_frontier.cpp)
    pybind11_add_module(_yamada_frontier_dense MODULE src/knotted_graph/invariants/yamada/_yamada_frontier_dense.cpp)
    pybind11_add_module(_yamada_factorized_frontier MODULE src/knotted_graph/invariants/yamada/_yamada_factorized_frontier.cpp)

    if(TARGET Boost::headers)
        target_link_libraries(_yamada_factorized_frontier PRIVATE Boost::headers)
    elseif(TARGET Boost::boost)
        target_link_libraries(_yamada_factorized_frontier PRIVATE Boost::boost)
    else()
        target_include_directories(_yamada_factorized_frontier PRIVATE ${Boost_INCLUDE_DIRS})
    endif()

    # _yamada_factorized_frontier is the sole production diagram evaluator.
    # The raw frontier kernels are retained only as independent exact-reference
    # implementations used by regression tests; they are never selected by the
    # public production dispatch.
    foreach(target _yamada_native _yamada_frontier _yamada_frontier_dense _yamada_factorized_frontier)
        target_compile_features(${target} PRIVATE cxx_std_17)
        if(MSVC)
            target_compile_options(${target} PRIVATE /O2 /DNDEBUG)
        else()
            target_compile_options(${target} PRIVATE -O3 -DNDEBUG)
        endif()
    endforeach()

    install(
        TARGETS _yamada_native _yamada_frontier _yamada_frontier_dense _yamada_factorized_frontier
        LIBRARY DESTINATION knotted_graph/invariants/yamada
        RUNTIME DESTINATION knotted_graph/invariants/yamada
    )
else()
    message(WARNING "No C++ compiler found; building KnottedGraph with Python Yamada fallback only")
endif()
