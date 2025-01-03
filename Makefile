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

# Optimization flags and warnings
CFLAGS      =  -O2 -Wall -mcpu=cell $(MACHDEP) $(INCLUDE)
CXXFLAGS    =  $(CFLAGS)

# Linker flags
LDFLAGS     =  $(MACHDEP) -Wl,-Map,$(notdir $@).map

# Required libraries
LIBS        :=  -lrsx -lgcm_sys -lio -lsysutil -lrt -llv2 -lm

# Source files
CFILES      :=  
CPPFILES    :=  llama_ps3.cpp \
                transformer.cpp \
                math_utils.cpp \
                memory_utils.cpp \
                sampler.cpp \
                tokenizer.cpp \
                rsxutil.cpp

ifneq ($(BUILD),$(notdir $(CURDIR)))

export OUTPUT    :=  $(CURDIR)/$(TARGET)

export VPATH    :=  $(foreach dir,$(SOURCE_DIRS),$(CURDIR)/$(dir)) \
                    $(foreach dir,$(DATA),$(CURDIR)/$(dir))

export DEPSDIR  :=  $(CURDIR)/$(BUILD)

export BUILDDIR :=  $(CURDIR)/$(BUILD)

# Object files
OFILES      :=  $(CPPFILES:.cpp=.o) $(CFILES:.c=.o)

# Choose compiler based on file types
export LD   :=  $(CXX)

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