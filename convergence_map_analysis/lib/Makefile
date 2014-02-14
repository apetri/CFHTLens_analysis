CC = clang
CXX = clang++

CFLAGS = -Wall $(INCLUDES)
CPPFLAGS = -Wall $(INCLUDES)

INCLUDES = -I../include -I/usr/local/include

LIBOBJS = coordinates.o differentials.o minkmom.o noise_generator.o noise_templates.o peaks.o power_spectrum.o smooth.o inout.o

libconvergence: $(LIBOBJS)
	ar -r libconvergence.a $(LIBOBJS)
	ranlib libconvergence.a

%.o: %.c ../include/coordinates.h ../include/differentials.h ../include/systematics.h ../include/inout.h
	$(CC) -c $(CFLAGS) $< -o $@

smooth.o: smooth.cpp

.PHONY: clean
clean:
	rm -rf *.o 

.PHONY: distclean
distclean:
	rm -rf *.o libconvergence.a
