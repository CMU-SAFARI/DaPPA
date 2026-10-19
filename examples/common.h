#ifndef DAPPA_EXAMPLES_COMMON_H
#define DAPPA_EXAMPLES_COMMON_H

#include <cstdlib>

// With DAPPA_EMIT_RECIPE or DAPPA_EMIT_BBOP set, execute() only writes the
// generated code and does not compute any results.
inline bool emitOnly()
{
    return std::getenv("DAPPA_EMIT_RECIPE") != nullptr || std::getenv("DAPPA_EMIT_BBOP") != nullptr;
}

#endif // DAPPA_EXAMPLES_COMMON_H
