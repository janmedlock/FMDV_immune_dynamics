NAME := monodromy
SO := $(shell python3-config --extension-suffix)
MODULE := $(NAME)$(SO)

CFLAGS := $(shell python3-config --cflags)
CFLAGS += -fPIC

LDFLAGS := $(shell python3-config --ldflags)
LDFLAGS += -shared

LDLIBS := $(shell python3-config --libs)
LDLIBS += -lblas

# Suppress warning.
CPPFLAGS := -DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION


default: $(MODULE)

%.c:	%.pyx
	cython3 $< -o $@

%$(SO):	%.o
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	$(RM) $(MODULE)
