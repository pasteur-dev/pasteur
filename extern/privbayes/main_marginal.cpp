#include <string>
#include <iostream>
#include <fstream>
using namespace std;

#include "methods.h"

double exp_marginal(table &tbl, base &base, int k)
{
	const vector<vector<int>> mrgs = tools::kway(k, tools::vectorize(tbl.dim));
	double err = 0.0;
	for (const vector<int> &mrg : mrgs)
		err += tools::TVD(tbl.getCounts(mrg), base.getCounts(mrg));
	return err / mrgs.size();
}

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

	char *domain = argv[3];
	char *data = argv[4];
	char *out = argv[5];

	random_device rd; // non-deterministic random engine
	engine eng(rd()); // deterministic engine with a random seed

	table tbl("../_data/" + dataset, true);
	vector<int> queries = {2, 3};

	for (double theta : thetas)
	{
		cout << "theta: " << theta << endl;
		out << "theta: " << theta << endl;
		for (double epsilon : {0.05, 0.1, 0.2, 0.4, 0.8, 1.6})
		{
			vector<double> err(queries.size(), 0.0);
			for (int i = 0; i < rep; i++)
			{
				cout << "epsilon: " << epsilon << " rep: " << i << endl;
				bayesian bayesian(eng, tbl, epsilon, theta);
				for (int qi = 0; qi < err.size(); qi++)
					err[qi] += exp_marginal(tbl, bayesian, queries[qi]);
			}
			for (int qi = 0; qi < err.size(); qi++)
				out << err[qi] / rep << "\t";
			out << endl;
		}
		cout << endl;
		out << endl;
	}
	out.close();
	log.close();
	return 0;
}