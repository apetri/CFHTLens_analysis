CC = clang
CFLAGS = -g -Wall $(INCLUDES)

INCLUDES = -I../include -I/usr/local/include

LIBOBJS = coordinates.o differentials.o minkmom.o noise_generator.o noise_templates.o peaks.o power_spectrum.o smooth.o

libconvergence: $(LIBOBJS)
	ar -r libconvergence.a $(LIBOBJS);
	ranlib libconvergence.a

%.o: %.c ../include/coordinates.h ../differentials.h ../systematics.h
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf *.o 

.PHONY: distclean
distclean:
	rm -rf *.o libconvergence.a
