CC=g++
CFLAGS= -I ./eigen/ -fopenmp
GDBDEBUGGER=-g
LDFLAGS=
SOURCES=cd_svm.cpp

OBJECTS=$(SOURCES:.cpp=.o)

EXECUTABLE=cd-svm

all: $(OBJECTS)
$(OBJECTS): $(SOURCES)
	$(CC) $(CFLAGS) $(GDBDEBUGGER) $(SOURCES) -o $(EXECUTABLE)

seq1:
	./cd-sq 0.1 16 ./data/covtype.tr ./data/covtype.t

seq2:   
	./cd-sq 0.1 16 ./data/rcv1.tr ./data/rcv1.t

atomic1:
	./cd-atomic 0.1 16 ./data/covtype.tr ./data/covtype.t

atomic2:
	./cd-atomic 0.1 16 ./data/rcv1.tr ./data/rcv1.t

asyn1:
	./cd-asyn 0.1 16 ./data/covtype.tr ./data/covtype.t

asyn2:
	./cd-asyn 0.1 16 ./data/rcv1.tr ./data/rcv1.t

test1:
	./cd-svm 0.1 16 ./data/covtype.tr ./data/covtype.t

test2:
	./cd-svm 0.1 16 ./data/rcv1.tr ./data/rcv1.t

clean:
	rm $(OBJECTS)
