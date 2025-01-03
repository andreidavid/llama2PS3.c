#include "tokenizer.h"

// PSL1GHT / PS3 includes
#include <ppu-lv2.h>
#include <sys/file.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

// -----------------------------------------------------------------------------
// 1) Helper for endianness
// -----------------------------------------------------------------------------
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
    memcpy(&value, &temp, sizeof(float));
    return value;
}

// -----------------------------------------------------------------------------
// 2) Compare function for sorting
// -----------------------------------------------------------------------------
int compare_tokens(const void *a, const void *b) {
    const TokenIndex* A = (const TokenIndex*)a;
    const TokenIndex* B = (const TokenIndex*)b;
    return strcmp(A->str, B->str);
}

// -----------------------------------------------------------------------------
// 3) build_tokenizer(): load from tokenizer.bin with PS3 syscalls + byteswap
// -----------------------------------------------------------------------------
void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    // Clear the struct
    memset(t, 0, sizeof(Tokenizer));
    t->vocab_size = vocab_size;

    // Allocate arrays
    t->vocab        = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;

    // Prepare single-byte pieces
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    // Open file via PS3 syscalls
    int fd;
    int ret = sysLv2FsOpen(tokenizer_path, SYS_O_RDONLY, &fd, 0, NULL, 0);
    if (ret != 0) {
        printf("Failed to open tokenizer file: %s (ret=%d)\n", tokenizer_path, ret);
        exit(EXIT_FAILURE);
    }

    uint64_t bytes_read;

    // 1) read max_token_length (an int), then byteswap
    unsigned int raw_len;
    ret = sysLv2FsRead(fd, &raw_len, sizeof(int), &bytes_read);
    if (ret != 0 || bytes_read != sizeof(int)) {
        printf("Failed to read max_token_length from %s\n", tokenizer_path);
        sysLv2FsClose(fd);
        exit(EXIT_FAILURE);
    }
    // convert from little-endian
    raw_len = (unsigned int)swap32((int32_t)raw_len);
    t->max_token_length = raw_len;

    // 2) read each vocab entry
    for (int i = 0; i < vocab_size; i++) {
        // read float vocab_score
        float raw_score;
        ret = sysLv2FsRead(fd, &raw_score, sizeof(float), &bytes_read);
        if (ret != 0 || bytes_read != sizeof(float)) {
            printf("Failed to read vocab_score for idx=%d\n", i);
            sysLv2FsClose(fd);
            exit(EXIT_FAILURE);
        }
        // byteswap
        float swapped_score = swap_float(raw_score);
        t->vocab_scores[i] = swapped_score;

        // read int token_len
        int raw_int;
        ret = sysLv2FsRead(fd, &raw_int, sizeof(int), &bytes_read);
        if (ret != 0 || bytes_read != sizeof(int)) {
            printf("Failed to read token_len for idx=%d\n", i);
            sysLv2FsClose(fd);
            exit(EXIT_FAILURE);
        }
        raw_int = swap32(raw_int);
        int token_len = raw_int;

        // allocate space for the token string
        t->vocab[i] = (char*)malloc(token_len + 1);
        // read raw bytes of token string
        ret = sysLv2FsRead(fd, t->vocab[i], token_len, &bytes_read);
        if (ret != 0 || bytes_read != (uint64_t)token_len) {
            printf("Failed to read token string for idx=%d\n", i);
            sysLv2FsClose(fd);
            exit(EXIT_FAILURE);
        }
        t->vocab[i][token_len] = '\0'; // add null terminator
    }

    // close file
    sysLv2FsClose(fd);
}

// -----------------------------------------------------------------------------
// 4) free_tokenizer()
// -----------------------------------------------------------------------------
void free_tokenizer(Tokenizer* t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            free(t->vocab[i]);
        }
        free(t->vocab);
    }
    if (t->vocab_scores) {
        free(t->vocab_scores);
    }
    if (t->sorted_vocab) {
        free(t->sorted_vocab);
    }
    memset(t, 0, sizeof(Tokenizer));
}

// -----------------------------------------------------------------------------
// 5) decode()
// -----------------------------------------------------------------------------
char* decode(Tokenizer* t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) {
        return (char*)""; 
    }
    char* piece = t->vocab[token];
    if (!piece) return (char*)"";

    // e.g. if piece is "<0x7F>", parse
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + (byte_val * 2);
    }

    // skip leading space if prev_token=1 (BOS)
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    return piece;
}

// -----------------------------------------------------------------------------
// 6) safe_printf()
// -----------------------------------------------------------------------------
void safe_printf(char *piece) {
    if (!piece || piece[0] == '\0') return;
    // if single char
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

// -----------------------------------------------------------------------------
// 7) str_lookup()
// -----------------------------------------------------------------------------
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex key;
    key.str = str;
    // bsearch
    TokenIndex* res = (TokenIndex*)bsearch(&key, sorted_vocab, vocab_size,
                                           sizeof(TokenIndex), compare_tokens);
    if (res) {
        return res->id;
    }
    return -1;
}

// -----------------------------------------------------------------------------
// 8) encode()
// -----------------------------------------------------------------------------
void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (!text) {
        fprintf(stderr, "Cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    // build sorted_vocab if needed
    if (!t->sorted_vocab) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id  = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // We'll store merges in a temp buffer
    char* str_buffer = (char*)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    *n_tokens = 0;

    // optional BOS=1
    if (bos) tokens[(*n_tokens)++] = 1;

    // add a dummy prefix if text != ""
    if (text[0] != '\0') {
        int dummy_prefix_id = str_lookup((char*)" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix_id < 0) dummy_prefix_id = 3; // fallback to <unk>=3
        tokens[(*n_tokens)++] = dummy_prefix_id;
    }

    // parse raw UTF-8 codepoints
    for (char *c = text; *c != '\0'; c++) {
        // if not continuation
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // if next is continuation, keep going if str_len < 4
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // we have a full codepoint in str_buffer
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // fallback: each byte => +3
            for (int i = 0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = ((unsigned char)str_buffer[i]) + 3;
            }
        }
        str_len = 0;
    }

    // BPE merges
    while (1) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int merged_id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (merged_id != -1) {
                float sc = t->vocab_scores[merged_id];
                if (sc > best_score) {
                    best_score = sc;
                    best_id = merged_id;
                    best_idx = i;
                }
            }
        }

        if (best_idx == -1) {
            break; // no merges found
        }
        // merge
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    // optional EOS=2
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}
