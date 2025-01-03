#ifndef __MEMORY_UTILS_H__
#define __MEMORY_UTILS_H__

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include "transformer.h"

/* PS3-specific memory allocation with 128-byte alignment */
void* ps3_malloc(size_t size);
void ps3_free(void* ptr);

/* PS3-specific endianness conversion helpers */
int32_t swap32(int32_t value);
float swap_float(float value);

/* Memory mapping and weight loading utilities 
 * Returns 1 on success, 0 on failure */
int read_ps3_checkpoint(const char* checkpoint_path, Config* config, size_t* mapped_size, char* error_message);
int load_ps3_weights(const char* checkpoint_path, Config* config, TransformerWeights* weights, char* error_message);

#endif /* __MEMORY_UTILS_H__ */