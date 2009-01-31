INCLUDE(CheckCSourceCompiles)
INCLUDE(CheckCXXSourceCompiles)
INCLUDE(CMakeBuildTypesList)

INCLUDE(AdvancedSet)
INCLUDE(AssertDefined)
INCLUDE(MultilineSet)
INCLUDE(DualScopeSet)


FUNCTION(CHECK_C_COMPILER_FLAGS FLAGS VAR)
  IF (NOT ${VAR} STREQUAL OFF)
    SET(CMAKE_REQUIRED_FLAGS ${FLAGS})
    CHECK_C_SOURCE_COMPILES("int main() {return 0;}" ${VAR})
    MARK_AS_ADVANCED(${VAR})
  ENDIF()
ENDFUNCTION()


FUNCTION(CHECK_CXX_COMPILER_FLAGS FLAGS VAR)
  IF (NOT ${VAR} STREQUAL OFF)
    SET(CMAKE_REQUIRED_FLAGS ${FLAGS})
    CHECK_CXX_SOURCE_COMPILES("int main() {return 0;}"  ${VAR})
    MARK_AS_ADVANCED(${VAR})
  ENDIF()
ENDFUNCTION()


MACRO(PACKAGE_ARCH_SET_LANGUAGE_BUILDTYPE_FLAGS LANG BUILDTYPE)

  IF (${PROJECT_NAME}_ENABLE_${LANG}_${BUILDTYPE}_COMPILE_FLAGS)
    DUAL_SCOPE_SET(CMAKE_${LANG}_FLAGS_${BUILDTYPE} ${GENERAL_${BUILDTYPE}_FLAGS})
    IF(${PROJECT_NAME}_VERBOSE_CONFIGURE)
      MESSAGE(STATUS "Adding ${LANG} ${BUILDTYPE} flags \"${GENERAL_${BUILDTYPE}_FLAGS}\"")
      PRINT_VAR(CMAKE_${LANG}_FLAGS_${BUILDTYPE})
    ENDIF()
  ENDIF()

ENDMACRO()


MACRO(PACKAGE_ARCH_SET_LANGUAGE_STRONG_WARNING_FLAGS LANG)

  ASSERT_DEFINED(${PROJECT_NAME}_ENABLE_STRONG_${LANG}_COMPILE_WARNINGS)
  IF (${PROJECT_NAME}_ENABLE_STRONG_${LANG}_COMPILE_WARNINGS)
    DUAL_SCOPE_SET(CMAKE_${LANG}_FLAGS
     "${${LANG}_STRONG_COMPILE_WARNING_FLAGS} ${CMAKE_${LANG}_FLAGS}")
    IF(${PROJECT_NAME}_VERBOSE_CONFIGURE)
      MESSAGE(STATUS "Adding strong ${LANG} warning flags \"${${LANG}_STRONG_COMPILE_WARNING_FLAGS}\"")
      PRINT_VAR(CMAKE_${LANG}_FLAGS)
    ENDIF()
  ENDIF()

ENDMACRO()


#
# Function that sets up strong compile options for the primary
# development platform (i.e. gcc)
#
# NOTE: The compiler flags in the cache, which may have been set by
# the user, are not disturbed in this function.  Instead variables in
# the parent base scope are set.
#

FUNCTION(PACKAGE_ARCH_SETUP_STRONG_COMPILE_WARNINGS)

  #
  # Setup and general flags
  #

  SET(GENERAL_DEBUG_FLAGS "-g -O0")

  SET(GENERAL_RELEASE_FLAGS "-O3")

  MULTILINE_SET(C_STRONG_COMPILE_WARNING_FLAGS
    " -ansi" # Check for C89 or C++98 standard code
    " -pedantic" # Adds more strick checking to remove non-ANSI GNU extensions
    " -Wall " # Enable a bunch of default warnings
    " -fexceptions" # Make sure that exceptions can be propogated through C code
    " -Wno-long-long" # Allow long long int since it is used by MPI, SWIG, etc.
    )
  
  MULTILINE_SET(CXX_STRONG_COMPILE_WARNING_FLAGS
    ${C_STRONG_COMPILE_WARNING_FLAGS}
    " -Wwrite-strings" # Checks for non-const char * copy of string constants
    " -Wshadow" # Warn about general shadowing issues
    " -Woverloaded-virtual" # Warn about hiding virtual functions
    )

  ADVANCED_SET( ${PROJECT_NAME}_WARNINGS_AS_ERRORS_FLAGS "-Werror"
    CACHE STRING "Flags for treating warnings as errors.  To turn off set to ''")

  #
  # C compiler options
  #

  ASSERT_DEFINED(${PROJECT_NAME}_ENABLE_C)
  IF (${PROJECT_NAME}_ENABLE_C)

    CHECK_C_COMPILER_FLAGS(${GENERAL_DEBUG_FLAGS}
      ${PROJECT_NAME}_ENABLE_C_DEBUG_COMPILE_FLAGS )
    PACKAGE_ARCH_SET_LANGUAGE_BUILDTYPE_FLAGS(C DEBUG)
    
    CHECK_C_COMPILER_FLAGS(${GENERAL_RELEASE_FLAGS}
      ${PROJECT_NAME}_ENABLE_C_RELEASE_COMPILE_FLAGS )
    PACKAGE_ARCH_SET_LANGUAGE_BUILDTYPE_FLAGS(C RELEASE)

    CHECK_C_COMPILER_FLAGS( ${C_STRONG_COMPILE_WARNING_FLAGS}
      ${PROJECT_NAME}_ENABLE_STRONG_C_COMPILE_WARNINGS )
    PACKAGE_ARCH_SET_LANGUAGE_STRONG_WARNING_FLAGS(C)
  
  ENDIF()

  #
  # C++ compiler options
  #

  ASSERT_DEFINED(${PROJECT_NAME}_ENABLE_CXX)
  IF (${PROJECT_NAME}_ENABLE_CXX)

    CHECK_CXX_COMPILER_FLAGS(${GENERAL_DEBUG_FLAGS}
      ${PROJECT_NAME}_ENABLE_CXX_DEBUG_COMPILE_FLAGS )
    PACKAGE_ARCH_SET_LANGUAGE_BUILDTYPE_FLAGS(CXX DEBUG)
    
    CHECK_CXX_COMPILER_FLAGS(${GENERAL_RELEASE_FLAGS}
      ${PROJECT_NAME}_ENABLE_CXX_RELEASE_COMPILE_FLAGS )
    PACKAGE_ARCH_SET_LANGUAGE_BUILDTYPE_FLAGS(CXX RELEASE)

    CHECK_CXX_COMPILER_FLAGS( ${CXX_STRONG_COMPILE_WARNING_FLAGS}
      ${PROJECT_NAME}_ENABLE_STRONG_CXX_COMPILE_WARNINGS )
    PACKAGE_ARCH_SET_LANGUAGE_STRONG_WARNING_FLAGS(CXX)
  
  ENDIF()
  
ENDFUNCTION()
