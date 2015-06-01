CC=g++
CFLAGS= -I ./eigen/ -fopenmp -std=c++0x -O3
GDBDEBUGGER=-g
LDFLAGS=
SOURCES=spMatrix.cpp newtonMethod.cpp

OBJECTS=$(SOURCES:.cpp=.o)

EXECUTABLE=nt-method

all: $(OBJECTS)
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(GDBDEBUGGER) $(SOURCES) -o $(EXECUTABLE)

clean:
	rm $(OBJECTS)
