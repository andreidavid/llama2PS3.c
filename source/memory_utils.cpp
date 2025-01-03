#include "memory_utils.h"
#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <ppu-lv2.h>
#include <sys/file.h>

void* ps3_malloc(size_t size) {
    void* ptr = memalign(128, size); // PS3 requires 128-byte alignment
    if (!ptr) {
        fprintf(stderr, "Failed to allocate %zu bytes\n", size);
        return NULL;
    }
    return ptr;
}

void ps3_free(void* ptr) {
    free(ptr);
}

int32_t swap32(int32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x000000FF) << 24);
}

float swap_float(float value) {
    uint32_t temp;
    memcpy(&temp, &value, sizeof(float));
    temp = swap32(temp);
    float result;
    memcpy(&result, &temp, sizeof(float));
    return result;
}

bool read_ps3_checkpoint(const char* checkpoint_path, Config* config, size_t* mapped_size) {
    int fd;
    uint64_t bytes_read;
    int32_t values[7];
    
    int ret = sysLv2FsOpen(checkpoint_path, SYS_O_RDONLY, &fd, 0, NULL, 0);
    if (ret != 0) {
        return false;
    }

    // Read config values
    ret = sysLv2FsRead(fd, values, 7 * sizeof(int32_t), &bytes_read);
    if (ret == 0 && bytes_read == 7 * sizeof(int32_t)) {
        // Swap endianness for each value
        config->dim         = swap32(values[0]);
        config->hidden_dim  = swap32(values[1]);
        config->n_layers    = swap32(values[2]);
        config->n_heads     = swap32(values[3]);
        config->n_kv_heads  = swap32(values[4]);
        config->vocab_size  = swap32(values[5]);
        config->seq_len     = swap32(values[6]);

        // Calculate weights size
        *mapped_size = (
            config->vocab_size * config->dim + // token embedding table
            config->n_layers * (
                config->dim + // rms_att_weight
                config->dim + // rms_ffn_weight
                config->dim * config->dim + // wq
                config->dim * config->dim + // wk
                config->dim * config->dim + // wv
                config->dim * config->dim + // wo
                config->dim * config->hidden_dim + // w1
                config->hidden_dim * config->dim + // w2
                config->dim * config->hidden_dim   // w3
            ) +
            config->dim // rms_final_weight
        ) * sizeof(float);

        sysLv2FsClose(fd);
        return true;
    }
    
    sysLv2FsClose(fd);
    return false;
}

bool load_ps3_weights(const char* checkpoint_path, Config* config, TransformerWeights* weights) {
    int fd;
    uint64_t bytes_read;
    int ret = sysLv2FsOpen(checkpoint_path, SYS_O_RDONLY, &fd, 0, NULL, 0);
    
    if (ret != 0) {
        return false;
    }

    // Skip the config header
    sysLv2FsLSeek64(fd, sizeof(Config), SEEK_SET, &bytes_read);

    // Calculate sizes
    size_t vocab_size = config->vocab_size;
    size_t dim = config->dim;
    size_t hidden_dim = config->hidden_dim;
    size_t n_layers = config->n_layers;

    // Allocate all weight buffers with PS3 alignment
    weights->token_embedding_table = (float*)ps3_malloc(vocab_size * dim * sizeof(float));
    weights->rms_att_weight       = (float*)ps3_malloc(n_layers * dim * sizeof(float));
    weights->rms_ffn_weight       = (float*)ps3_malloc(n_layers * dim * sizeof(float));
    weights->wq                   = (float*)ps3_malloc(n_layers * dim * dim * sizeof(float));
    weights->wk                   = (float*)ps3_malloc(n_layers * dim * dim * sizeof(float));
    weights->wv                   = (float*)ps3_malloc(n_layers * dim * dim * sizeof(float));
    weights->wo                   = (float*)ps3_malloc(n_layers * dim * dim * sizeof(float));
    weights->w1                   = (float*)ps3_malloc(n_layers * dim * hidden_dim * sizeof(float));
    weights->w2                   = (float*)ps3_malloc(n_layers * hidden_dim * dim * sizeof(float));
    weights->w3                   = (float*)ps3_malloc(n_layers * dim * hidden_dim * sizeof(float));
    weights->rms_final_weight     = (float*)ps3_malloc(dim * sizeof(float));

    // Helper to read and byteswap weights
    auto read_weights = [&](float* ptr, size_t count) {
        ret = sysLv2FsRead(fd, ptr, count * sizeof(float), &bytes_read);
        if (ret == 0 && bytes_read == count * sizeof(float)) {
            for (size_t i = 0; i < count; i++) {
                ptr[i] = swap_float(ptr[i]);
            }
            return true;
        }
        return false;
    };

    // Read all weights with endianness conversion
    bool success = true;
    success &= read_weights(weights->token_embedding_table, vocab_size * dim);
    success &= read_weights(weights->rms_att_weight,       n_layers * dim);
    success &= read_weights(weights->wq,                   n_layers * dim * dim);
    success &= read_weights(weights->wk,                   n_layers * dim * dim);
    success &= read_weights(weights->wv,                   n_layers * dim * dim);
    success &= read_weights(weights->wo,                   n_layers * dim * dim);
    success &= read_weights(weights->rms_ffn_weight,       n_layers * dim);
    success &= read_weights(weights->w1,                   n_layers * dim * hidden_dim);
    success &= read_weights(weights->w2,                   n_layers * hidden_dim * dim);
    success &= read_weights(weights->w3,                   n_layers * dim * hidden_dim);
    success &= read_weights(weights->rms_final_weight,     dim);

    sysLv2FsClose(fd);
    return success;
}