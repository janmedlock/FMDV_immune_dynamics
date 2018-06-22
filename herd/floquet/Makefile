NAME := _monodromy

EXT += $(shell python3-config --extension-suffix)
CFLAGS := $(shell python3-config --cflags)
CFLAGS += -fPIC
LDFLAGS := $(shell python3-config --ldflags)

MODULE := $(NAME)$(EXT)


default: $(MODULE)

%.c:	%.pyx
	cython $< -o $@

%$(EXT):	%.o
	$(CC) $(LDFLAGS) -shared $^ $(LDLIBS) -o $@

clean:
	$(RM) $(MODULE)
