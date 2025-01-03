#ifndef __SAMPLER_H__
#define __SAMPLER_H__

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;  // buffer for top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n);
int sample_mult(float* probabilities, int n, float coin);
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin);
int sample(Sampler* sampler, float* logits);

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler* sampler);

#endif