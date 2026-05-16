function(add_green_dependency TARGET)
    include(FetchContent)
    FetchContent_Declare(
        ${TARGET}
        GIT_REPOSITORY https://github.com/Green-Phys/${TARGET}.git
        GIT_TAG origin/main
    )
    FetchContent_MakeAvailable(${TARGET})
endfunction()
