CC=icc
CFLAGS= -fopenmp -std=c++0x -O3
GDBDEBUGGER=-g
LDFLAGS=
SOURCES=spMatrix.cpp pagerank.cpp

OBJECTS=$(SOURCES:.cpp=.o)

EXECUTABLE=pagerank.out

all: $(OBJECTS)
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(GDBDEBUGGER) $(SOURCES) -o $(EXECUTABLE)

clean:
	rm $(OBJECTS)
