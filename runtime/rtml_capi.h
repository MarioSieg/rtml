// Copyright Mario "Neo" Sieg 2024. All rights reserved. mario.sieg.64@gmail.com

#ifndef RTML_CAPI_H
#define RTML_CAPI_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSVC_VER
#define RTML_API __declspec(dllexport)
#else
#define RTML_API __attribute__((visibility("default")))
#endif

extern RTML_API bool rtml_global_init(void);
extern RTML_API void rtml_global_shutdown(void);

typedef uint32_t rtml_tensor_id_t;
extern RTML_API void rtml_isolate_create(
   const char* name,
   uint32_t /*context::compute_device*/ device,
   size_t memory_budged
);
extern RTML_API bool rtml_isolate_exists(const char* name);
extern RTML_API rtml_tensor_id_t rtml_isolate_create_tensor(
   const char* isolate_name,
   uint32_t /*tensor::stype*/ data_type,
   int64_t d1, int64_t d2,
   int64_t d3, int64_t d4,
   uint32_t shape_len,
   rtml_tensor_id_t view,
   size_t slice_offset
);
extern RTML_API const char* rtml_tensor_print(const char* isolate_name, rtml_tensor_id_t id);

#ifdef __cplusplus
}
#endif

#endif
