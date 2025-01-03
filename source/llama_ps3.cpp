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

static vs32 dialog_action = 0;

static void dialog_handler(msgButton button, void *usrData) {
    dialog_action = 1;
}

static void do_flip() {
    sysUtilCheckCallback();
    flip();
}

static void show_message(const char* message) {
    msgType dialogType = (msgType)(MSG_DIALOG_NORMAL | MSG_DIALOG_BTN_OK);
    msgDialogOpen2(dialogType, message, dialog_handler, NULL, NULL);
}

void test_generate() {
    char message[2048] = {0};
    char temp[256] = {0};
    Transformer transformer = {0};
    Tokenizer tokenizer = {0};
    Sampler sampler = {0};
    bool success = true;

    // Build the transformer
    build_transformer(&transformer, (char*)"/dev_usb006/PS3/USRDIR/stories15M.bin");

    // Build the tokenizer
    build_tokenizer(&tokenizer, "/dev_usb006/PS3/USRDIR/tokenizer.bin", transformer.config.vocab_size);

    // Build the sampler (temperature = 1.0, topp = 0.9)
    build_sampler(&sampler, transformer.config.vocab_size, 1.0f, 0.9f, 1234ull);

    // Set up the prompt
    const char* prompt = "Sleepy Joe said";
    strcat(message, "Prompt: \"");
    strcat(message, prompt);
    strcat(message, "\"\n");

    // Encode the prompt
    int prompt_tokens[512];  // large enough buffer for the prompt
    int n_prompt_tokens = 0;
    encode(&tokenizer, (char*)prompt, /*bos=*/1, /*eos=*/0, prompt_tokens, &n_prompt_tokens);

    if (n_prompt_tokens < 1) {
        strcat(message, "Error: No valid tokens in prompt!\n");
        success = false;
    }

    // Generation loop
    if (success) {
        int steps = 60;  // number of tokens to generate
        int pos = 0;     // position in sequence
        int token = prompt_tokens[0];  // start with first token of prompt
        
        strcat(message, "\nGenerated text: ");

        while (pos < steps) {
            // Forward pass to get logits
            float* logits = forward(&transformer, token, pos);
            
            // Choose next token
            int next;
            if (pos < n_prompt_tokens - 1) {
                // Still processing the prompt, use the next prompt token
                next = prompt_tokens[pos + 1];
            } else {
                // Sample from the logits to get the next token
                next = sample(&sampler, logits);
            }

            pos++;

            // Stop if we see special end tokens
            if (next == 1 || next == 2) {  // BOS or EOS
                sprintf(temp, "\nEncountered special token (%d). Stopping.\n", next);
                strcat(message, temp);
                break;
            }

            // Decode the token to text and append to output
            char* piece = decode(&tokenizer, token, next);
            if (piece && strlen(piece) + strlen(message) < sizeof(message) - 100) {  // leave room for final messages
                strcat(message, piece);
            }

            token = next;
        }

        strcat(message, "\n\nGeneration complete.\n");
    }

    // Clean up in reverse order of initialization
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);

    // Show results
    show_message(message);
}

// Main function with basic UI setup
int main(int argc, char *argv[]) {
    void *host_addr = memalign(1024*1024, HOST_SIZE);
    init_screen(host_addr, HOST_SIZE);
    
    // Run the text generation
    test_generate();

    // Wait for dialog
    dialog_action = 0;
    for(int i = 0; i < 1000 && !dialog_action; i++) {
        do_flip();
        usleep(20000);
    }

    msgDialogAbort();
    
    // Return to PS3Load
    sysProcessExitSpawn2("/dev_hdd0/game/PSL145310/RELOAD.SELF", NULL, NULL, NULL, 0, 1001, SYS_PROCESS_SPAWN_STACK_SIZE_1M);
    
    return 0;
}