#ifndef __RSXUTIL_H__
#define __RSXUTIL_H__

#include <ppu-types.h>
#include <rsx/rsx.h>

/* Constants for RSX initialization */
#define CB_SIZE         0x100000
#define HOST_SIZE       (32*1024*1024)

/* Global variables needed by the RSX */
extern gcmContextData *context;
extern u32 display_width;
extern u32 display_height;
extern u32 curr_fb;

/* Core RSX functions */
void setRenderTarget(u32 index);
void init_screen(void *host_addr, u32 size);
void waitflip(void);
void flip(void);

#endif /* __RSXUTIL_H__ */