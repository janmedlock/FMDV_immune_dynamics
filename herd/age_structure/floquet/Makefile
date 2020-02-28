NAME := monodromy
EXT += $(shell python3-config --extension-suffix)
MODULE := $(NAME)$(EXT)

# Suppress NPY_NO_DEPRECATED_API #warning.
CPPFLAGS := -Wno-cpp

CFLAGS := $(shell python3-config --cflags)
CFLAGS += -fPIC

LDFLAGS := $(shell python3-config --ldflags)
LDFLAGS += -lblas


default: $(MODULE)

%.c:	%.pyx
	cython3 $< -o $@

%$(EXT):	%.o
	$(CC) $(LDFLAGS) -shared $^ $(LDLIBS) -o $@

clean:
	$(RM) $(MODULE)
