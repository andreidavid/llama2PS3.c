#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include <stdint.h>

/* struct used in sorted vocab, represents one token and its id */
typedef struct {
    char *str;
    int id;
} TokenIndex;

/* the tokenizer struct */
typedef struct {
    char** vocab;           /* vocabulary strings */
    float* vocab_scores;    /* vocabulary scores */
    TokenIndex* sorted_vocab; /* vocab sorted for binary search */
    int vocab_size;         /* vocabulary size */
    unsigned int max_token_length; /* max token length from tokenizer.bin */
    unsigned char byte_pieces[512]; /* individual byte tokens */
} Tokenizer;

/* Compare function for qsort() and bsearch() usage */
int compare_tokens(const void *a, const void *b);

/* Build tokenizer from file with PS3 byte handling */
void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);

/* Free the memory */
void free_tokenizer(Tokenizer* t);

/* BOS=1, EOS=2 token ids. Returns number of tokens encoded in tokens[] array */
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);

/* Convert token id back to string */
char* decode(Tokenizer* t, int prev_token, int token);

/* Safe print piece ensuring no control chars */
void safe_printf(char *piece);

#endif /* __TOKENIZER_H__ */