CC := clang

#CFLAGS := -w -g -O3 -ffp-contract=on -fopenmp
CFLAGS = -w -g -fopenmp -O3 -ffp-contract=on -mcpu=native -mtune=native -march=armv8.2-a+sve -ftree-vectorize -pthread 
#-Rpass-analysis=loop-vectorize


LDFLAGS = -fopenmp

SRC = $(wildcard *.c)

OBJ = $(patsubst %.c, %.o, ${SRC})

EXEC = ./polynomial_stencil

all: $(EXEC)

$(EXEC): $(OBJ)
		$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c
		$(CC) $(CFLAGS) -c $< -o $@

clean:
		rm -f $(OBJ) $(EXEC)
