#include "transformer.h"
#include "math_utils.h"
#include <malloc.h>
#include <string.h>
#include <stdio.h>
#include <ppu-lv2.h>
#include <sys/file.h>

/* PS3-specific memory alignment requirement */
static void* malloc_aligned(size_t size) {
    return memalign(128, size); /* 128-byte alignment for PS3 */
}

static void free_aligned(void* ptr) {
    free(ptr);
}

void malloc_run_state(RunState* s, Config* p) {
    /* Calculate dimensions */
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    /* Allocate all buffers with PS3 alignment */
    s->x = (float*)malloc_aligned(p->dim * sizeof(float));
    s->xb = (float*)malloc_aligned(p->dim * sizeof(float));
    s->xb2 = (float*)malloc_aligned(p->dim * sizeof(float));
    s->hb = (float*)malloc_aligned(p->hidden_dim * sizeof(float));
    s->hb2 = (float*)malloc_aligned(p->hidden_dim * sizeof(float));
    s->q = (float*)malloc_aligned(p->dim * sizeof(float));
    s->k = (float*)malloc_aligned(p->dim * sizeof(float));
    s->v = (float*)malloc_aligned(p->dim * sizeof(float));
    s->att = (float*)malloc_aligned(p->n_heads * p->seq_len * sizeof(float));
    s->logits = (float*)malloc_aligned(p->vocab_size * sizeof(float));
    s->key_cache = (float*)malloc_aligned(p->n_layers * p->seq_len * kv_dim * sizeof(float));
    s->value_cache = (float*)malloc_aligned(p->n_layers * p->seq_len * kv_dim * sizeof(float));

    /* Initialize key and value cache to zeros */
    if (s->key_cache) {
        memset(s->key_cache, 0, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    }
    if (s->value_cache) {
        memset(s->value_cache, 0, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    }

    /* Validate allocations */
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v ||
        !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free_aligned(s->x);
    free_aligned(s->xb);
    free_aligned(s->xb2);
    free_aligned(s->hb);
    free_aligned(s->hb2);
    free_aligned(s->q);
    free_aligned(s->k);
    free_aligned(s->v);
    free_aligned(s->att);
    free_aligned(s->logits);
    free_aligned(s->key_cache);
    free_aligned(s->value_cache);
}

/* Helper function for PS3 endianness handling */
static int32_t swap32(int32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x000000FF) << 24);
}

static float swap_float(float value) {
    uint32_t temp;
    memcpy(&temp, &value, sizeof(float));
    temp = swap32(temp);
    float result;
    memcpy(&result, &temp, sizeof(float));
    return result;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                    int* fd, float** data, ssize_t* file_size) {
    /* Open using PS3 syscall */
    uint64_t bytes_read;
    int ret = sysLv2FsOpen(checkpoint, SYS_O_RDONLY, fd, 0, NULL, 0);
    if (ret != 0) {
        fprintf(stderr, "Failed to open checkpoint file\n");
        exit(EXIT_FAILURE);
    }

    /* Read config with endianness conversion */
    int32_t raw_values[7];
    ret = sysLv2FsRead(*fd, raw_values, sizeof(raw_values), &bytes_read);
    if (ret == 0 && bytes_read == sizeof(raw_values)) {
        config->dim = swap32(raw_values[0]);
        config->hidden_dim = swap32(raw_values[1]);
        config->n_layers = swap32(raw_values[2]);
        config->n_heads = swap32(raw_values[3]);
        config->n_kv_heads = swap32(raw_values[4]);
        config->vocab_size = swap32(raw_values[5]);
        config->seq_len = swap32(raw_values[6]);
    } else {
        fprintf(stderr, "Failed to read config\n");
        exit(EXIT_FAILURE);
    }

    /* Calculate file size */
    uint64_t pos;
    sysLv2FsLSeek64(*fd, 0, SEEK_END, &pos);
    *file_size = pos;
    sysLv2FsLSeek64(*fd, 0, SEEK_SET, &pos);

    /* Allocate memory for the entire file */
    *data = (float*)malloc_aligned(*file_size);
    if (!*data) {
        fprintf(stderr, "Failed to allocate memory for checkpoint\n");
        exit(EXIT_FAILURE);
    }

    /* Read the entire file */
    ret = sysLv2FsRead(*fd, *data, *file_size, &bytes_read);
    if (ret != 0 || bytes_read != (uint64_t)*file_size) {
        fprintf(stderr, "Failed to read checkpoint data\n");
        exit(EXIT_FAILURE);
    }

    /* Set up weight pointers (skipping config at start) */
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    
    /* Map the weights following run.c pattern */
    int head_size = config->dim / config->n_heads;
    weights->token_embedding_table = weights_ptr;
    weights_ptr += config->vocab_size * config->dim;
    
    weights->rms_att_weight = weights_ptr;
    weights_ptr += config->n_layers * config->dim;
    
    weights->wq = weights_ptr;
    weights_ptr += config->n_layers * config->dim * config->dim;
    
    weights->wk = weights_ptr;
    weights_ptr += config->n_layers * config->dim * (config->n_kv_heads * head_size);
    
    weights->wv = weights_ptr;
    weights_ptr += config->n_layers * config->dim * (config->n_kv_heads * head_size);
    
    weights->wo = weights_ptr;
    weights_ptr += config->n_layers * (config->n_heads * head_size) * config->dim;
    
    weights->rms_ffn_weight = weights_ptr;
    weights_ptr += config->n_layers * config->dim;
    
    weights->w1 = weights_ptr;
    weights_ptr += config->n_layers * config->dim * config->hidden_dim;
    
    weights->w2 = weights_ptr;
    weights_ptr += config->n_layers * config->hidden_dim * config->dim;
    
    weights->w3 = weights_ptr;
    weights_ptr += config->n_layers * config->dim * config->hidden_dim;
    
    weights->rms_final_weight = weights_ptr;
    
    /* Handle endianness for all float values */
    size_t float_count = *file_size / sizeof(float);
    size_t i;
    for (i = 0; i < float_count; i++) {
        (*data)[i] = swap_float((*data)[i]);
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    /* Call the core forward implementation */
    forward_impl(&transformer->config, &transformer->weights, &transformer->state, token, pos);
    return transformer->state.logits;
}

void build_transformer(Transformer* t, char* checkpoint_path) {
    /* Zero out the transformer struct */
    memset(t, 0, sizeof(Transformer));
    
    /* Read in the config and weights */
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    
    /* Allocate the run state buffers */
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    /* Free the run state */
    free_run_state(&t->state);
    
    /* Free the mapped data */
    if (t->data) {
        free_aligned(t->data);
    }
    
    /* Close file descriptor */
    if (t->fd != -1) {
        sysLv2FsClose(t->fd);
    }
    
    /* Zero out the struct */
    memset(t, 0, sizeof(Transformer));
}