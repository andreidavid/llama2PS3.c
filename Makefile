.SUFFIXES:

include $(PSL1GHT)/ppu_rules

TARGET      :=  $(notdir $(CURDIR))
BUILD       :=  build
SOURCE_DIRS :=  source
DATA        :=  data
INCLUDES    :=  include source

TITLE       :=  LLama PS3
APPID       :=  LLAMA001
CONTENTID   :=  UP0001-$(APPID)_00-0000000000000000

# C specific flags
CFLAGS      =  -std=gnu89 -O2 -Wall -mcpu=cell $(MACHDEP) $(INCLUDE)
# Add debug info flags if needed
#CFLAGS     +=  -g

# Linker flags
LDFLAGS     =  $(MACHDEP) -Wl,-Map,$(notdir $@).map

# Required libraries
LIBS        :=  -lrsx -lgcm_sys -lio -lsysutil -lrt -llv2 -lm

# Source files
CFILES      :=  llama_ps3.c \
                transformer.c \
                math_utils.c \
                memory_utils.c \
                sampler.c \
                tokenizer.c \
                rsxutil.c

ifneq ($(BUILD),$(notdir $(CURDIR)))

export OUTPUT    :=  $(CURDIR)/$(TARGET)

export VPATH    :=  $(foreach dir,$(SOURCE_DIRS),$(CURDIR)/$(dir)) \
                    $(foreach dir,$(DATA),$(CURDIR)/$(dir))

export DEPSDIR  :=  $(CURDIR)/$(BUILD)

export BUILDDIR :=  $(CURDIR)/$(BUILD)

# Object files
OFILES      :=  $(CFILES:.c=.o)

# Choose C compiler
export LD   :=  $(CC)

export OFILES    :=  $(OFILES)

# Include paths
export INCLUDE   :=  $(foreach dir,$(INCLUDES),-I$(CURDIR)/$(dir)) \
                     $(foreach dir,$(LIBDIRS),-I$(dir)/include) \
                     $(LIBPSL1GHT_INC) \
                     -I$(CURDIR)/$(BUILD)

# Library paths                    
export LIBPATHS  :=  $(foreach dir,$(LIBDIRS),-L$(dir)/lib) \
                     $(LIBPSL1GHT_LIB)

.PHONY: $(BUILD) clean

# Build rules
$(BUILD):
	@[ -d $@ ] || mkdir -p $@
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile

clean:
	@echo cleaning ...
	@rm -fr $(BUILD) $(OUTPUT).elf $(OUTPUT).self

else

DEPENDS :=  $(OFILES:.o=.d)

$(OUTPUT).self: $(OUTPUT).elf
$(OUTPUT).elf:  $(OFILES)

-include $(DEPENDS)

endif