CC := clang

#CFLAGS := -w -g -O3 -ffp-contract=on -fopenmp
CFLAGS = -I/home/PAC20248205/software/new_numactl_install/include -w -g -fopenmp -O3 -ffp-contract=on -mcpu=native -mtune=native -march=armv8.2-a+sve -ftree-vectorize -pthread
#-Rpass-analysis=loop-vectorize

LDFLAGS = -L/home/PAC20248205/software/new_numactl_install/lib -fopenmp -lnuma -Wl,--wrap=malloc

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

