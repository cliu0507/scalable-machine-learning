CC=icc
CFLAGS= -fopenmp -std=c++0x -O3
GDBDEBUGGER=-g
LDFLAGS=
SOURCES=spMatrix.cpp powerIter.cpp

OBJECTS=$(SOURCES:.cpp=.o)

EXECUTABLE=p-Iter

all: $(OBJECTS)
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(GDBDEBUGGER) $(SOURCES) -o $(EXECUTABLE)

test1:
	./sp-mult 16 meta_data.txt data.txt

gdb1:
	gdb --args ./sp-mult 16 meta_data.txt data.txt

valgrind1:
	valgrind ./sp-mult 16 meta_data.txt data.txt

test2:
	./p-Iter 16 meta_data.txt data.txt

gdb2:
	gdb --args ./p-Iter 16 meta_data.txt data.txt

valgrind2:
	valgrind ./p-Iter 16 meta_data.txt data.txt

clean:
	rm $(OBJECTS)
