#ifndef __TRANSFORMER_H__
#define __TRANSFORMER_H__

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

/* Configuration structure from run.c */
typedef struct {
    int dim;        /* transformer dimension */
    int hidden_dim; /* for ffn layers */
    int n_layers;   /* number of layers */
    int n_heads;    /* number of query heads */
    int n_kv_heads; /* number of key/value heads (can be < query heads because of multiquery) */
    int vocab_size; /* vocabulary size */
    int seq_len;    /* max sequence length */
} Config;

/* Weights for the transformer */
typedef struct {
    /* token embedding table */
    float* token_embedding_table;    /* (vocab_size, dim) */
    /* weights for rmsnorms */
    float* rms_att_weight;          /* (layer, dim) rmsnorm weights */
    float* rms_ffn_weight;          /* (layer, dim) */
    /* weights for matmuls. note dim == n_heads * head_size */
    float* wq;                      /* (layer, dim, n_heads * head_size) */
    float* wk;                      /* (layer, dim, n_kv_heads * head_size) */
    float* wv;                      /* (layer, dim, n_kv_heads * head_size) */
    float* wo;                      /* (layer, n_heads * head_size, dim) */
    /* weights for ffn */
    float* w1;                      /* (layer, hidden_dim, dim) */
    float* w2;                      /* (layer, dim, hidden_dim) */
    float* w3;                      /* (layer, hidden_dim, dim) */
    /* final rmsnorm */
    float* rms_final_weight;        /* (dim,) */
    /* (optional) classifier weights for the logits, on the last layer */
    float* wcls;
} TransformerWeights;

/* RunState for the forward pass */
typedef struct {
    /* current wave of activations */
    float* x;         /* activation at current time stamp (dim,) */
    float* xb;        /* same, but inside a residual branch (dim,) */
    float* xb2;       /* an additional buffer just for convenience (dim,) */
    float* hb;        /* buffer for hidden dimension in the ffn (hidden_dim,) */
    float* hb2;       /* buffer for hidden dimension in the ffn (hidden_dim,) */
    float* q;         /* query (dim,) */
    float* k;         /* key (dim,) */
    float* v;         /* value (dim,) */
    float* att;       /* buffer for scores/attention values (n_heads, seq_len) */
    float* logits;    /* output logits */
    /* kv cache */
    float* key_cache;   /* (layer, seq_len, dim) */
    float* value_cache; /* (layer, seq_len, dim) */
} RunState;

/* The transformer struct that combines everything */
typedef struct {
    Config config;           /* the hyperparameters of the architecture */
    TransformerWeights weights; /* the weights of the model */
    RunState state;         /* buffers for the "wave" of activations in the forward pass */
    /* some more state needed to properly clean up the memory mapping */
    int fd;                 /* file descriptor for memory mapping */
    float* data;           /* memory mapped data pointer */
    ssize_t file_size;     /* size of the checkpoint file in bytes */
} Transformer;

/* Core functions matching run.c signatures */
void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);
float* forward(Transformer* transformer, int token, int pos);
void build_transformer(Transformer* t, char* checkpoint_path);
void free_transformer(Transformer* t);

/* Memory mapping functions */
void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                    int* fd, float** data, ssize_t* file_size);

#endif /* __TRANSFORMER_H__ */