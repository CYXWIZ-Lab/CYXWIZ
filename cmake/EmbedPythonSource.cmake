# Turns a Python file into a C++ include holding its text, so pycyxwiz can
# install it as a submodule without shipping a separate .py (TOFIX125 P3,
# Part C6 option A).
#
#   cmake -DINPUT=<file.py> -DOUTPUT=<file.inc> -DNAME=<identifier> -P EmbedPythonSource.cmake
#
# The output defines `static const char* const NAME[]` (pieces of the text,
# ending with nullptr): MSVC limits one string literal to about 16 KB.

file(READ "${INPUT}" _text)
string(FIND "${_text}" ")PYSRC\"" _clash)
if(NOT _clash EQUAL -1)
    message(FATAL_ERROR "${INPUT} contains the raw-string delimiter )PYSRC\"")
endif()

set(_out "// Generated from ${INPUT} by EmbedPythonSource.cmake. Do not edit.\n")
string(APPEND _out "static const char* const ${NAME}[] = {\n")
string(LENGTH "${_text}" _length)
set(_chunk 12000)
set(_offset 0)
while(_offset LESS _length)
    string(SUBSTRING "${_text}" ${_offset} ${_chunk} _piece)
    string(APPEND _out "R\"PYSRC(${_piece})PYSRC\",\n")
    math(EXPR _offset "${_offset} + ${_chunk}")
endwhile()
string(APPEND _out "nullptr};\n")
file(WRITE "${OUTPUT}" "${_out}")
