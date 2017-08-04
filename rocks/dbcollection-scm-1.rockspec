package = "dbcollection"
version = "scm-1"

source = {
    url = "git://github.com/dbcollection/dbcollection-torch7.git",
    tag = "master"
 }

description = {
    summary = "A Lua/Torch7 wrapper for dbcollection.",
    detailed = [[
        This is a simple Lua wrapper for the Python's dbcollection module. The functionality
        is almost the same, appart from some few minor differences related to Lua, namely
        regarding setting up ranges when fetching data.

        Internally it calls the Python's dbcollection module for data download/process/management.
        The, Lua/Torch7 interacts solely with the metadata hdf5 file to fetch data from disk.
    ]],
    homepage = "https://github.com/dbcollection/dbcollection-torch7",
    license = "MIT",
    maintainer = "M. Farrajota"
 }

dependencies = {
    "lua >= 5.1",
    "torch >= 7.0",
    "json >= 1.0",
    "hdf5 >= 20-0",
    "argcheck >= 1.0",
    "dok >= scm-1"
}

build = {
    type = "cmake",
    variables = {
        CMAKE_BUILD_TYPE="Release",
        LUA_PATH="$(LUADIR)",
        LUA_CPATH="$(LIBDIR)"
   }
}