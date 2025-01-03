#ifndef __SAMPLER_H__
#define __SAMPLER_H__

/* struct used when sorting probabilities during top-p sampling */
typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;  /* buffer for top-p sampling */
    float temperature;     /* temperature for sampling */
    float topp;           /* top-p sampling threshold */
    unsigned long long rng_state; /* random number generator state */
} Sampler;

/* Core sampling functions */
int sample_argmax(float* probabilities, int n);
int sample_mult(float* probabilities, int n, float coin);
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin);
int sample(Sampler* sampler, float* logits);

/* Sampler initialization and cleanup */
void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);

#endif /* __SAMPLER_H__ */