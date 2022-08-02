all: privbayes.bin

%.bin: *.cpp
	g++ -O3 --std=c++14 $^ -o $@
	chmod +x $@

clean:
	rm *.bin