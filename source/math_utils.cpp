#include "math_utils.h"
#include <math.h>
#include <string.h>

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void forward_impl(Config* config, TransformerWeights* weights, RunState* state, int token, int pos) {
    // a few convenience variables
    float *x = state->x;
    int dim = config->dim;
    int kv_dim = (config->dim * config->n_kv_heads) / config->n_heads;
    int kv_mul = config->n_heads / config->n_kv_heads; // integer multiplier of the kv sharing
    int hidden_dim = config->hidden_dim;
    int head_size = dim / config->n_heads;

    // copy the token embedding into x
    float* content_row = weights->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(*x));

    // forward all the layers
    for(int l = 0; l < config->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(state->xb, x, weights->rms_att_weight + l*dim, dim);

        // compute key and value vectors for this position
        int loff = l * config->seq_len * kv_dim; // kv cache layer offset
        float* key_cache_row = state->key_cache + loff + pos * kv_dim;
        float* value_cache_row = state->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(state->q, state->xb, weights->wq + l*dim*dim, dim, dim);
        matmul(key_cache_row, state->xb, weights->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(value_cache_row, state->xb, weights->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? state->q : key_cache_row;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        for (int h = 0; h < config->n_heads; h++) {
            // get the query vector for this head
            float* q = state->q + h * head_size;
            // attention scores for this head
            float* att = state->att + h * config->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = state->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights
            softmax(att, pos + 1);

            // weighted sum of the values, store into xb
            float* xb = state->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = state->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(state->xb2, state->xb, weights->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += state->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(state->xb, x, weights->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        matmul(state->hb, state->xb, weights->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(state->hb2, state->xb, weights->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = state->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= state->hb2[i];
            state->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(state->xb, state->hb, weights->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += state->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, weights->rms_final_weight, dim);

    // classifier into logits
    matmul(state->logits, x, weights->token_embedding_table, dim, config->vocab_size);
}