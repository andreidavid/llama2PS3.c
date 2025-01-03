#ifndef __MATH_UTILS_H__
#define __MATH_UTILS_H__

#include "transformer.h"

/* Core math functions copied from run.c */
void rmsnorm(float* o, float* x, float* weight, int size);
void softmax(float* x, int size);
void matmul(float* xout, float* x, float* w, int n, int d);

/* Internal implementation of the forward pass */
void forward_impl(Config* config, TransformerWeights* weights, RunState* state, int token, int pos);

#endif /* __MATH_UTILS_H__ */