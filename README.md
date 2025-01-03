# LLama2 on PlayStation 3

This project ports [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy to run on the PlayStation 3's Cell Broadband Engine processor. It demonstrates running a small language model on 2006-era console hardware.

## Features
- Runs the stories15M model (15M parameters) on PS3
- Pure C implementation optimized for PowerPC architecture
- Handles PS3's big-endian memory requirements
- Memory-aligned data structures for Cell processor

## Requirements
- PlayStation 3 with custom firmware (e.g., CFW, mmCM)
- stories15M.bin model file (~60MB)
- PSL1GHT SDK
- PS3 Toolchain

## Building
1. Set up the PS3 development environment:
```bash
export PS3DEV=/usr/local/ps3dev
export PSL1GHT=$PS3DEV
export PATH=$PATH:$PS3DEV/bin
export PATH=$PATH:$PS3DEV/ppu/bin
export PATH=$PATH:$PS3DEV/spu/bin
```

2. Clone and build:
```bash
git clone [https://github.com/andreidavid/llama2PS3.c.git]
cd llama-ps3
make
```

3. Install the model:
```bash
# Download the model file
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

4. Transfer files to PS3:
   - Copy stories15M.bin and tokenizer.bin to a USB drive
   - Create directory PS3/USRDIR/ on the USB drive
   - Place stories15M.bin and tokenizer.bin in the USRDIR directory
   - Insert USB drive into PS3

## Running
1. Load ps3load on your PS3
2. From your development machine:
```bash
export PS3LOAD=tcp:<your-ps3-ip>
ps3load hello_world.self
```

## Technical Details

### Hardware Utilization
- CPU: Cell Broadband Engine (3.2GHz PowerPC + 8 SPEs)
- RAM: 256MB XDR DRAM
- Model Size: ~60MB for stories15M
- Memory Alignment: 128-byte alignment required for Cell processor

### Architecture Considerations
- Big-endian architecture (requires byte swapping for model weights)
- Strict memory alignment requirements
- Potential for SPE parallelization (future optimization)

### Memory Management
- Custom memory allocator with 128-byte alignment
- Explicit endianness handling for model weights
- Careful pointer management for struct fields

## Current Status
- [x] Pure C implementation
- [x] Basic model loading and validation
- [x] Forward pass implementation
- [x] Memory management optimized for PS3
- [x] Sampling implementation
- [x] Basic inference working
- [x] Token generation
- [ ] SPE optimization
- [ ] Interactive chat mode
- [ ] Performance optimization

## Known Issues
- Limited by PowerPC core performance
- SPE acceleration not yet implemented
- Memory usage could be optimized further

## Future Improvements
- Implement SPE acceleration for matrix multiplication
- Add quantization support
- Optimize memory usage
- Add streaming support for larger models

## Contributing
Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## License
[License information]

## Acknowledgments
- Andrej Karpathy for the original llama2.c implementation
- PSL1GHT developers for the PS3 SDK
- PS3 homebrew community

## References
- [llama2.c](https://github.com/karpathy/llama2.c)
- [PSL1GHT SDK](https://github.com/ps3dev/PSL1GHT)
- [PS3 Toolchain](https://github.com/ps3dev/ps3toolchain)