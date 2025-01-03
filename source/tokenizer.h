#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include <stdint.h> // for int8_t

// This struct holds one token string + its ID (used in sorted_vocab)
typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    // The raw vocabulary strings
    char** vocab;
    // A float score for each vocab entry
    float* vocab_scores;
    // A sorted array of TokenIndex, for merges / str lookups
    TokenIndex* sorted_vocab;
    // total number of tokens in the vocabulary
    int vocab_size;
    // maximum token length in characters (from tokenizer.bin)
    unsigned int max_token_length;
    // for raw byte tokens, e.g. <0xXX>
    unsigned char byte_pieces[512];
} Tokenizer;

// Compare function for qsort() and bsearch() usage
int compare_tokens(const void *a, const void *b);

// Build the tokenizer from tokenizer.bin (PS3 syscalls, plus byteswapping)
void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size);

// Release memory
void free_tokenizer(Tokenizer* t);

// Convert (prev_token, token) -> string piece (handles <0xXX> raw bytes)
char* decode(Tokenizer* t, int prev_token, int token);

// Print only safe/printable text
void safe_printf(char *piece);

// Look up a string in the sorted vocab. returns token ID or -1
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);

// Encode a UTF-8 string into tokens. Optionally BOS=1, EOS=2
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens);

#endif // __TOKENIZER_H__
