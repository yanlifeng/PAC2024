CC := clang

#CFLAGS := -w -g -O3 -ffp-contract=on -fopenmp
CFLAGS = -I/home/PAC20248205/numactl/include -w -g -fopenmp -O3 -ffp-contract=on -mcpu=native -mtune=native -march=armv8.2-a+sve -ftree-vectorize -pthread -lnuma
#-Rpass-analysis=loop-vectorize


LDFLAGS = -L/home/PAC20248205/numactl/lib -fopenmp -lnuma

SRC = $(wildcard *.c)

OBJ = $(patsubst %.c, %.o, ${SRC})

EXEC = ./polynomial_stencil

all: $(EXEC)

$(EXEC): $(OBJ)
		$(CC) $^ -o $@  $(LDFLAGS) 

%.o: %.c
		$(CC) $(CFLAGS) -c $< -o $@

clean:
		rm -f $(OBJ) $(EXEC)
