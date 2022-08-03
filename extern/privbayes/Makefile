all: privbayes.bin

%.bin: *.cpp *.h
	g++ -O3 --std=c++14 $(filter %.cpp,$^) -o $@
	chmod +x $@

clean:
	rm *.bin

.PHONY: all clean run