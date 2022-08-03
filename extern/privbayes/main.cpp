#include <string>
#include <iostream>
#include <fstream>
using namespace std;

#include "methods.h"

int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		cerr << "Format: ./privbayes.bin <e> <beta> <theta> <domain> <data> <out>" << endl;
		return -1;
	}

	double e = stod(argv[0]);
	double beta = stod(argv[1]);
	double theta = stod(argv[2]);

	string domain = argv[3];
	string data = argv[4];
	string out = argv[5];

	return 0;
}