all: privbayes.bin

%.bin: *.cpp *.h
	g++ -O3 --std=c++14 $^ -o $@
	chmod +x $@

clean:
	rm *.bin

.PHONY: all clean run