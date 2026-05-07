include_guard(GLOBAL)

set(_PHIRST_INTEGRAND_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(_phirst_integrand_resolve_path out_var input_path)
    if(IS_ABSOLUTE "${input_path}")
        set(_resolved "${input_path}")
    else()
        set(_resolved "${CMAKE_CURRENT_SOURCE_DIR}/${input_path}")
    endif()

    if(NOT EXISTS "${_resolved}")
        message(FATAL_ERROR "phirst_add_integrand_module: path does not exist: ${input_path}")
    endif()

    set(${out_var} "${_resolved}" PARENT_SCOPE)
endfunction()

function(_phirst_integrand_resolve_include_dir out_var)
    if(DEFINED PHIRST_INCLUDE_DIR AND EXISTS "${PHIRST_INCLUDE_DIR}")
        set(${out_var} "${PHIRST_INCLUDE_DIR}" PARENT_SCOPE)
        return()
    endif()

    get_filename_component(_phirst_source_root "${_PHIRST_INTEGRAND_MODULE_DIR}/.." ABSOLUTE)
    set(_source_include_dir "${_phirst_source_root}/include")
    if(EXISTS "${_source_include_dir}/phirst/phirst.hpp")
        set(${out_var} "${_source_include_dir}" PARENT_SCOPE)
        return()
    endif()

    message(FATAL_ERROR
        "phirst_add_integrand_module: could not locate phirst headers. "
        "Use find_package(phirst REQUIRED) or include the helper from the phirst source tree.")
endfunction()

function(_phirst_integrand_link_target_if_present target)
    foreach(_candidate IN LISTS ARGN)
        if(TARGET "${_candidate}")
            target_link_libraries(${target} PRIVATE "${_candidate}")
            return()
        endif()
    endforeach()
endfunction()

function(_phirst_integrand_configure_backend target backend)
    string(TOUPPER "${backend}" _backend)

    if(_backend STREQUAL "SERIAL")
        _phirst_integrand_link_target_if_present(${target} phirst::serial phirst::phirst_serial phirst_serial)
        if(NOT TARGET phirst::serial AND NOT TARGET phirst::phirst_serial AND NOT TARGET phirst_serial)
            _phirst_integrand_resolve_include_dir(_phirst_include_dir)
            target_include_directories(${target} PRIVATE "${_phirst_include_dir}")
            target_compile_definitions(${target} PRIVATE PHIRST_BACKEND_SERIAL)
        endif()
        return()
    endif()

    if(_backend STREQUAL "KOKKOS")
        if(NOT TARGET Kokkos::kokkos)
            find_package(Kokkos REQUIRED)
        endif()
        _phirst_integrand_link_target_if_present(${target} phirst::kokkos phirst_kokkos)
        if(NOT TARGET phirst::kokkos AND NOT TARGET phirst_kokkos)
            _phirst_integrand_resolve_include_dir(_phirst_include_dir)
            target_include_directories(${target} PRIVATE "${_phirst_include_dir}")
            target_link_libraries(${target} PRIVATE Kokkos::kokkos)
            target_compile_definitions(${target} PRIVATE PHIRST_BACKEND_KOKKOS)
        endif()
        return()
    endif()

    if(_backend STREQUAL "CUDA")
        include(CheckLanguage)
        check_language(CUDA)
        if(NOT CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR "phirst_add_integrand_module: CUDA backend requested but no CUDA compiler was found.")
        endif()
        enable_language(CUDA)

        get_target_property(_target_sources ${target} SOURCES)
        if(_target_sources)
            set_source_files_properties(${_target_sources} PROPERTIES LANGUAGE CUDA)
        endif()

        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )

        _phirst_integrand_link_target_if_present(${target} phirst::cuda phirst_cuda)
        if(NOT TARGET phirst::cuda AND NOT TARGET phirst_cuda)
            _phirst_integrand_resolve_include_dir(_phirst_include_dir)
            target_include_directories(${target} PRIVATE "${_phirst_include_dir}")
            target_compile_definitions(${target} PRIVATE PHIRST_BACKEND_CUDA)
            target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
        endif()
        return()
    endif()

    message(FATAL_ERROR "phirst_add_integrand_module: unsupported BACKEND '${backend}'. Valid values: SERIAL, CUDA, KOKKOS")
endfunction()

function(phirst_add_integrand_module)
    set(options)
    set(oneValueArgs NAME SOURCE INTEGRAND_TYPE PARTICLES BACKEND HEADER)
    cmake_parse_arguments(PHIRST_MODULE "${options}" "${oneValueArgs}" "" ${ARGN})

    foreach(_required IN ITEMS NAME SOURCE INTEGRAND_TYPE PARTICLES)
        if(NOT PHIRST_MODULE_${_required})
            message(FATAL_ERROR "phirst_add_integrand_module: missing required argument ${_required}")
        endif()
    endforeach()

    if(NOT PHIRST_MODULE_BACKEND)
        set(PHIRST_MODULE_BACKEND SERIAL)
    endif()

    string(TOUPPER "${PHIRST_MODULE_BACKEND}" _backend)
    if(NOT _backend MATCHES "^(SERIAL|CUDA|KOKKOS)$")
        message(FATAL_ERROR "phirst_add_integrand_module: BACKEND must be SERIAL, CUDA, or KOKKOS")
    endif()

    if(NOT PHIRST_MODULE_PARTICLES MATCHES "^[2-6]$")
        message(FATAL_ERROR "phirst_add_integrand_module: PARTICLES must be an integer in the range 2..6")
    endif()

    _phirst_integrand_resolve_path(_user_source "${PHIRST_MODULE_SOURCE}")

    if(PHIRST_MODULE_HEADER)
        _phirst_integrand_resolve_path(_user_header "${PHIRST_MODULE_HEADER}")
        set(_wrapper_include_path "${_user_header}")
        set(_module_sources "${_user_source}")
    else()
        set(_wrapper_include_path "${_user_source}")
        set(_module_sources)
    endif()

    string(REPLACE "\\" "/" _wrapper_include_path "${_wrapper_include_path}")
    set(USER_INTEGRAND_INCLUDE_LINE "#include \"${_wrapper_include_path}\"")
    set(USER_INTEGRAND_TYPE "${PHIRST_MODULE_INTEGRAND_TYPE}")
    set(PHIRST_N_PARTICLES "${PHIRST_MODULE_PARTICLES}")
    set(PHIRST_BACKEND_DEFINE "PHIRST_BACKEND_${_backend}")

    set(_wrapper_source "${CMAKE_CURRENT_BINARY_DIR}/phirst_wrapper_${PHIRST_MODULE_NAME}.cpp")
    configure_file(
        "${_PHIRST_INTEGRAND_MODULE_DIR}/integrand_wrapper_template.cpp.in"
        "${_wrapper_source}"
        @ONLY
    )

    list(APPEND _module_sources "${_wrapper_source}")

    add_library(${PHIRST_MODULE_NAME} SHARED ${_module_sources})
    set_target_properties(${PHIRST_MODULE_NAME} PROPERTIES
        OUTPUT_NAME "${PHIRST_MODULE_NAME}"
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )

    target_compile_definitions(${PHIRST_MODULE_NAME} PRIVATE
        PHIRST_N_PARTICLES=${PHIRST_MODULE_PARTICLES}
        PHIRST_BACKEND_${_backend}
    )

    _phirst_integrand_configure_backend(${PHIRST_MODULE_NAME} "${_backend}")
endfunction()
