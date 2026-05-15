function(add_green_dependency TARGET)
    include(FetchContent)
    # Don't build the green dep's own tests when consuming it.
    set(Build_Tests OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        ${TARGET}
        GIT_REPOSITORY https://github.com/Green-Phys/${TARGET}.git
        GIT_TAG origin/main
    )
    FetchContent_MakeAvailable(${TARGET})
endfunction()
