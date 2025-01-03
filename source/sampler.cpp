#include "sampler.h"
#include "math_utils.h"
#include "memory_utils.h"

#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <malloc.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ppu-lv2.h>
#include <sys/file.h>

unsigned int random_u32(unsigned long long *state) {
   *state ^= *state >> 12;
   *state ^= *state << 25;
   *state ^= *state >> 27;
   return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
   return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_argmax(float* probabilities, int n) {
   int max_i = 0;
   float max_p = probabilities[0];
   for (int i = 1; i < n; i++) {
       if (probabilities[i] > max_p) {
           max_i = i;
           max_p = probabilities[i];
       }
   }
   return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
   float cdf = 0.0f;
   for (int i = 0; i < n; i++) {
       cdf += probabilities[i];
       if (coin < cdf) {
           return i;
       }
   }
   return n - 1;
}

int compare(const void* a, const void* b) {
   ProbIndex* a_ = (ProbIndex*) a;
   ProbIndex* b_ = (ProbIndex*) b;
   if (a_->prob > b_->prob) return -1;
   if (a_->prob < b_->prob) return 1;
   return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
   int n0 = 0;
   const float cutoff = (1.0f - topp) / (n - 1);
   for (int i = 0; i < n; i++) {
       if (probabilities[i] >= cutoff) {
           probindex[n0].index = i;
           probindex[n0].prob = probabilities[i];
           n0++;
       }
   }
   qsort(probindex, n0, sizeof(ProbIndex), compare);

   float cumulative_prob = 0.0f;
   int last_idx = n0 - 1;
   for (int i = 0; i < n0; i++) {
       cumulative_prob += probindex[i].prob;
       if (cumulative_prob > topp) {
           last_idx = i;
           break;
       }
   }

   float r = coin * cumulative_prob;
   float cdf = 0.0f;
   for (int i = 0; i <= last_idx; i++) {
       cdf += probindex[i].prob;
       if (r < cdf) {
           return probindex[i].index;
       }
   }
   return probindex[last_idx].index;
}

int sample(Sampler* sampler, float* logits) {
   if (sampler->temperature == 0.0f) {
       return sample_argmax(logits, sampler->vocab_size);
   } else {
       for (int q=0; q<sampler->vocab_size; q++) {
           logits[q] /= sampler->temperature;
       }
       softmax(logits, sampler->vocab_size);
       float coin = random_f32(&sampler->rng_state);
       if (sampler->topp <= 0 || sampler->topp >= 1) {
           return sample_mult(logits, sampler->vocab_size, coin);
       } else {
           return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
       }
   }
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
   sampler->vocab_size = vocab_size;
   sampler->temperature = temperature;
   sampler->topp = topp;
   sampler->rng_state = rng_seed;
   sampler->probindex = (ProbIndex*)ps3_malloc(vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
   if (sampler->probindex) ps3_free(sampler->probindex);
   memset(sampler, 0, sizeof(Sampler));
}