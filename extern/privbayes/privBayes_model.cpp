#include <random>
#include <iostream>
#include <cmath>
using namespace std;
#include <stdio.h>
#include "methods.h"
#include "noise.h"
#include "privBayes_model.h"

string c_get_model(const int *data, int m, int n, const string &config, double e1, double e2, double theta, int seed)
{
	engine eng(seed); // deterministic engine with a random seed

	// Vanilla privbayes implementation from the paper
	table tbl(data, config, true, m, n);
	bayesian bayesian1(eng, tbl, e1, e2, theta);
	string m1 = bayesian1.print_model();
	return m1;
}
