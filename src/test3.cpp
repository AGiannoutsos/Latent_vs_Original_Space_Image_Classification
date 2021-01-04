#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <fstream>

#include "../include/pandac.h"
// #include "../include/numc.h"
// #include "../include/prediction_results.h"
// #include "../include/exhaustive_knn.h"
// #include "../include/lsh_classifier.h"


using namespace std;

int main(int argc, char** argv) {
	// Read query file with PandaC.
    NumC<int>* queryData = PandaC<int>::fromMNISTnew(argv[1], 10);
    cout << "+++++++++++++++++++++++++++++" << endl;
    cout << "END OF MINST" << endl;
    cout << "+++++++++++++++++++++++++++++" << endl;
    queryData->print();
    return 0;
}