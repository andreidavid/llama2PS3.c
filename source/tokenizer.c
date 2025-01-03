#include "tokenizer.h"
#include "memory_utils.h"
#include <ppu-lv2.h>
#include <sys/file.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* comparison function for qsort and bsearch */
int compare_tokens(const void *a, const void *b) {
    const TokenIndex* a_ = (const TokenIndex*)a;
    const TokenIndex* b_ = (const TokenIndex*)b;
    return strcmp(a_->str, b_->str);
}

static int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    /* efficiently find the perfect match for str in vocab, return its index or -1 if not found */
    TokenIndex tok = { .str = str }; /* acts as the key to search for */
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    unsigned int raw_len;
    int i;
    int fd;
    uint64_t bytes_read;
    float score;
    int len;
    int ret;

    /* clear the struct */
    memset(t, 0, sizeof(Tokenizer));
    t->vocab_size = vocab_size;

    /* allocate space for vocabulary and scores */
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; /* initialized lazily */

    /* init individual byte pieces */
    for (i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    /* open file using PS3 syscalls */
    ret = sysLv2FsOpen(tokenizer_path, SYS_O_RDONLY, &fd, 0, NULL, 0);
    if (ret != 0) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }

    /* read in max_token_length */
    ret = sysLv2FsRead(fd, &raw_len, sizeof(int), &bytes_read);
    if (ret != 0 || bytes_read != sizeof(int)) {
        fprintf(stderr, "failed to read max_token_length\n");
        exit(EXIT_FAILURE);
    }
    /* convert from little-endian */
    t->max_token_length = (unsigned int)swap32((int32_t)raw_len);

    /* read in all the vocabulary data */
    for (i = 0; i < vocab_size; i++) {
        /* read float score */
        ret = sysLv2FsRead(fd, &score, sizeof(float), &bytes_read);
        if (ret != 0 || bytes_read != sizeof(float)) {
            fprintf(stderr, "failed to read score\n");
            exit(EXIT_FAILURE);
        }
        t->vocab_scores[i] = swap_float(score);

        /* read token length */
        ret = sysLv2FsRead(fd, &len, sizeof(int), &bytes_read);
        if (ret != 0 || bytes_read != sizeof(int)) {
            fprintf(stderr, "failed to read len\n");
            exit(EXIT_FAILURE);
        }
        len = swap32(len);

        /* read the token string data */
        t->vocab[i] = (char*)malloc(len + 1);
        ret = sysLv2FsRead(fd, t->vocab[i], len, &bytes_read);
        if (ret != 0 || bytes_read != (uint64_t)len) {
            fprintf(stderr, "failed to read token string\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0'; /* add null terminator */
    }

    sysLv2FsClose(fd);
}

void free_tokenizer(Tokenizer* t) {
    int i;
    if (t->vocab) {
        for (i = 0; i < t->vocab_size; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    if (t->vocab_scores) free(t->vocab_scores);
    if (t->sorted_vocab) free(t->sorted_vocab);
    memset(t, 0, sizeof(Tokenizer));
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece;
    unsigned char byte_val;
    
    if (token < 0 || token >= t->vocab_size) {
        return "";
    }

    piece = t->vocab[token];
    if (!piece) {
        return "";
    }

    /* following BOS (1) token, sentencepiece decoder strips any leading whitespace */
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }

    /* handle the special byte pieces */
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }

    return piece;
}

void safe_printf(char *piece) {
    /* piece might be a raw byte token, and we only want to print printable chars or whitespace */
    if (piece == NULL) return;
    if (piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    char *str_buffer;
    char *c;
    size_t str_len;
    int i;
    int dummy_prefix;
    int id;
    float best_score;
    int best_id;
    int best_idx;
    
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    /* lazy initialize the sorted vocabulary */
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    /* create a temporary buffer that will store merged tokens */
    /* *2 for concat, +1 for null terminator +2 for UTF8 */
    str_buffer = malloc((t->max_token_length*2 + 1 + 2) * sizeof(char));
    str_len = 0;

    /* start at 0 tokens */
    *n_tokens = 0;

    /* add optional BOS (=1) token */
    if (bos) tokens[(*n_tokens)++] = 1;

    /* add_dummy_prefix is true by default */
    if (text[0] != '\0') {
        dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        if (dummy_prefix == -1) dummy_prefix = 3;  /* 3 is <unk> */
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    /* process the raw byte sequence of the input string */
    for (c = text; *c != '\0'; c++) {
        /* reset buffer if the current byte is ASCII or a leading byte */
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        /* append the current byte to the buffer */
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        /* if the next character is a continuation byte and str_len < 4, continue */
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        /* otherwise we have a full codepoint, so look it up in vocab */
        id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            /* byte fallback: just encode each byte as a token */
            for (i = 0; i < (int)str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    /* merge tokens based on scores as long as possible */
    while (1) {
        best_score = -1e10;
        best_id = -1;
        best_idx = -1;

        for (i = 0; i < (*n_tokens-1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;  /* we couldn't find any more tokens to merge */
        }

        /* merge the consecutive pair (best_idx, best_idx+1) into new token best_id */
        tokens[best_idx] = best_id;
        /* delete token at position best_idx+1, shift the entire sequence back 1 */
        for (i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    /* add optional EOS (=2) token */
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}