cmake_minimum_required(VERSION 3.15)
project(KnottedGraphNative LANGUAGES NONE)

# Keep source installations usable even on machines without a C++ compiler:
# the Python package has an exact arbitrary-precision fallback.
include(CheckLanguage)
check_language(CXX)

if(CMAKE_CXX_COMPILER)
    enable_language(CXX)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(
        _yamada_native
        MODULE
        src/knotted_graph/invariants/yamada/_yamada_native.cpp
    )
    pybind11_add_module(
        _yamada_iso
        MODULE
        src/knotted_graph/invariants/yamada/_yamada_iso.cpp
    )
    pybind11_add_module(
        _yamada_frontier
        MODULE
        src/knotted_graph/invariants/yamada/_yamada_frontier.cpp
    )
    target_compile_features(_yamada_native PRIVATE cxx_std_17)
    target_compile_features(_yamada_iso PRIVATE cxx_std_17)
    target_compile_features(_yamada_frontier PRIVATE cxx_std_17)

    if(MSVC)
        target_compile_options(_yamada_native PRIVATE /O2 /DNDEBUG)
        target_compile_options(_yamada_iso PRIVATE /O2 /DNDEBUG)
        target_compile_options(_yamada_frontier PRIVATE /O2 /DNDEBUG)
    else()
        target_compile_options(_yamada_native PRIVATE -O3 -DNDEBUG)
        target_compile_options(_yamada_iso PRIVATE -O3 -DNDEBUG)
        target_compile_options(_yamada_frontier PRIVATE -O3 -DNDEBUG)
    endif()

    install(
        TARGETS _yamada_native _yamada_iso _yamada_frontier
        LIBRARY DESTINATION knotted_graph/invariants/yamada
        RUNTIME DESTINATION knotted_graph/invariants/yamada
    )
else()
    message(WARNING "No C++ compiler found; building KnottedGraph with Python Yamada fallback only")
endif()
