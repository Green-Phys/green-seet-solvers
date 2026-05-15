function(add_edlib_dependency)
    include(FetchContent)
    # EDLib must build without ALPSCore — we use green-h5pp / green-params
    # instead. cpp-arnoldi is pulled in transitively via EDLib.
    set(EDLIB_WITH_ALPSCORE OFF CACHE BOOL "" FORCE)
    # Don't build EDLib's own tests / examples when consuming it.
    set(Testing  OFF CACHE BOOL "" FORCE)
    set(Examples OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        EDLib
        GIT_REPOSITORY https://github.com/Q-Solvers/EDLib.git
        GIT_TAG origin/remove-alpscore-dep
    )

    FetchContent_MakeAvailable(EDLib)
endfunction()
