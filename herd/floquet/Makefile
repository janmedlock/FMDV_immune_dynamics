# Build the Cython monodromy solver.

NAME := monodromy
SO := $(shell python3-config --extension-suffix)
MODULE := $(NAME)$(SO)

CFLAGS += $(shell python3-config --cflags)
CFLAGS += $(shell pkg-config numpy --cflags)
CFLAGS += -fPIC

LDLIBS += $(shell python3-config --libs)
LDLIBS += $(shell pkg-config numpy --libs)
LDLIBS += -lblas

LDFLAGS += $(shell python3-config --ldflags)
LDFLAGS += -shared

CPPFLAGS += -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION


default: $(MODULE)

%.c:	%.pyx
	cython3 $< -o $@

%$(SO):	%.o
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	$(RM) $(MODULE)
