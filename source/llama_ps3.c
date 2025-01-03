#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <malloc.h>
#include <sysutil/msg.h>
#include <sysutil/sysutil.h>
#include <sys/process.h>
#include "transformer.h"
#include "tokenizer.h"
#include "sampler.h"
#include "rsxutil.h"

/* Global variables for UI control */
static vs32 dialog_action = 0;

/* Callback for message dialog */
static void dialog_handler(msgButton button, void *usrData) {
    dialog_action = 1;
}

/* Screen update helper */
static void do_flip(void) {
    sysUtilCheckCallback();
    flip();
}

/* Main generation function */
void test_generate(void) {
    static char display_buffer[2048];
    Transformer transformer = {0};
    Tokenizer tokenizer = {0};
    Sampler sampler = {0};
    msgType dialogType;
    const char* prompt = "Once upon a time";
    int prompt_tokens[512];  /* large enough buffer for the prompt */
    int n_prompt_tokens = 0;
    int steps = 50;         /* number of tokens to generate */
    int pos = 0;            /* position in sequence */
    int token;              /* current token */
    int next;              /* next token */
    int success = 1;
    char* piece;

    /* Clear display buffer */
    display_buffer[0] = '\0';

    /* Initialize all components */
    build_transformer(&transformer, (char*)"/dev_usb006/PS3/USRDIR/stories15M.bin");
    build_tokenizer(&tokenizer, "/dev_usb006/PS3/USRDIR/tokenizer.bin", transformer.config.vocab_size);
    build_sampler(&sampler, transformer.config.vocab_size, 1.0f, 0.9f, 1234ull);

    /* Add initial text to buffer */
    strcat(display_buffer, "Prompt: \"");
    strcat(display_buffer, prompt);
    strcat(display_buffer, "\"\n\nGenerating: ");
    
    /* Show initial state */
    dialogType = (msgType)(MSG_DIALOG_NORMAL);
    msgDialogOpen2(dialogType, display_buffer, dialog_handler, NULL, NULL);
    do_flip();

    /* Encode the prompt */
    encode(&tokenizer, (char*)prompt, 1, 0, prompt_tokens, &n_prompt_tokens);

    if (n_prompt_tokens < 1) {
        strcat(display_buffer, "\nError: No valid tokens in prompt!\n");
        msgDialogOpen2(dialogType, display_buffer, dialog_handler, NULL, NULL);
        success = 0;
    }

    if (success) {
        float* logits;
        token = prompt_tokens[0];

        while (pos < steps) {
            logits = forward(&transformer, token, pos);
            
            if (pos < n_prompt_tokens - 1) {
                next = prompt_tokens[pos + 1];
            } else {
                next = sample(&sampler, logits);
            }

            pos++;

            if (next == 1 || next == 2) {  /* BOS or EOS */
                break;
            }

            /* Decode token and update display */
            piece = decode(&tokenizer, token, next);
            if (piece && strlen(piece) + strlen(display_buffer) < sizeof(display_buffer) - 100) {
                strcat(display_buffer, piece);
                msgDialogClose(0.0f);
                msgDialogOpen2(dialogType, display_buffer, dialog_handler, NULL, NULL);
                do_flip();
            }

            token = next;
        }

        /* Show completion */
        strcat(display_buffer, "\n\nGeneration complete.");
        msgDialogClose(0.0f);
        msgDialogOpen2((msgType)(MSG_DIALOG_NORMAL | MSG_DIALOG_BTN_TYPE_OK), 
                      display_buffer, dialog_handler, NULL, NULL);
    }

    /* Clean up in reverse order */
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
}

/* Program exit callback */
static void program_exit_callback(void) {
    gcmSetWaitFlip(context);
    rsxFinish(context, 1);
}

/* Main function */
int main(int argc, char *argv[]) {
    void *host_addr;
    int i;  /* Declare loop variable at start of block */

    host_addr = memalign(1024*1024, HOST_SIZE);
    init_screen(host_addr, HOST_SIZE);
    
    /* Register exit callback */
    atexit(program_exit_callback);

    /* Run the text generation */
    test_generate();

    /* Wait for dialog */
    dialog_action = 0;
    for(i = 0; i < 1000 && !dialog_action; i++) {
        do_flip();
        usleep(20000);
    }

    msgDialogAbort();
    
    /* Return to PS3Load */
    sysProcessExitSpawn2("/dev_hdd0/game/PSL145310/RELOAD.SELF", NULL, NULL, NULL, 0, 1001, SYS_PROCESS_SPAWN_STACK_SIZE_1M);
    
    return 0;
}