CC=icc
CFLAGS= -I ./eigen/ -fopenmp
GDBDEBUGGER=-g
LDFLAGS=
SOURCES=omp_als.cpp

OBJECTS=$(SOURCES:.cpp=.o)

EXECUTABLE=omp-als

all: $(OBJECTS)
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(GDBDEBUGGER) $(SOURCES) -o $(EXECUTABLE)

test: $(EXECUTABLE)
	./omp-als 20 1 16 ./data/large/
clean:
	rm $(OBJECTS)
