NAME := _monodromy
EXT += $(shell python3-config --extension-suffix)
MODULE := $(NAME)$(EXT)

CFLAGS := $(shell python3-config --cflags)
CFLAGS += -fPIC

LDFLAGS := $(shell python3-config --ldflags)
LDFLAGS += -lblas


default: $(MODULE)

%.c:	%.pyx
	cython $< -o $@

%$(EXT):	%.o
	$(CC) $(LDFLAGS) -shared $^ $(LDLIBS) -o $@

clean:
	$(RM) $(MODULE)
