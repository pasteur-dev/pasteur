#include <string>
#include <iostream>
#include <fstream>
using namespace std;

#include "privBayes_model.h"

int main(int argc, char *argv[])
{
	if (argc < 5)
	{
		cerr << "Format: ./privbayes.bin <m> <n> <e1> <e2> <theta> <seed> <domain>" << endl;
		return -1;
	}

	int i = 0;
	double m = stod(argv[i++]);
	double n = stod(argv[i++]);
	double e1 = stod(argv[i++]);
	double e2 = stod(argv[i++]);
	double theta = stod(argv[i++]);
	int seed = stoi(argv[i++]);

	string domain = argv[i++];

	string model = c_get_model(0, m, n, domain, e1, e2, theta, seed);
	cout << model;
	return 0;
}