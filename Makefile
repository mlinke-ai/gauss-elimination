.PHONY: avx base clean

base:
	gcc -fPIC -shared $(shell pkg-config --cflags --libs python3) $(shell numpy-config --cflags) -o cgauss.so cgauss.c

avx:
	gcc -fPIC -shared $(shell pkg-config --cflags --libs python3) $(shell numpy-config --cflags) -mavx2 -o cgauss.so cgauss.c

clean:
	rm cgauss.so
