#pragma once

// GCC raises some optimiser warnings (-Warray-bounds, -Wuninitialized) inside
// library headers once their code is inlined into ours; -isystem does not stop
// warnings raised that late. Wrap only the function or #include that reaches
// such a false positive, never a whole file (TOFIX187):
//
//   CYXWIZ_GCC_IGNORE_BEGIN("-Warray-bounds")
//   ...
//   CYXWIZ_GCC_IGNORE_END
#if defined(__GNUC__) && !defined(__clang__)
#define CYXWIZ_GCC_PRAGMA(x) _Pragma(#x)
#define CYXWIZ_GCC_IGNORE_BEGIN(warning) \
    _Pragma("GCC diagnostic push") CYXWIZ_GCC_PRAGMA(GCC diagnostic ignored warning)
#define CYXWIZ_GCC_IGNORE_END _Pragma("GCC diagnostic pop")
#else
#define CYXWIZ_GCC_IGNORE_BEGIN(warning)
#define CYXWIZ_GCC_IGNORE_END
#endif
