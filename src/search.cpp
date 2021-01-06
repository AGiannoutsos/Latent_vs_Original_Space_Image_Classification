// ./bin/search -d ./data/train-images-idx3-ubyte -i ./d1.txt -q ./data/t10k-images-idx3-ubyte -s ./q1.txt -o ./test_output.txt -k 4 -L 5
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <fstream>

#include "../include/pandac.h"
#include "../include/numc.h"
#include "../include/prediction_results.h"
#include "../include/exhaustive_knn.h"
#include "../include/lsh_classifier.h"

using namespace std;

// Extracts the results to output file.
bool extractResults(char* outputFile, Results* results_original, Results* results_reduced, Results* true_results, NumC<int>* original_inputData, NumC<int>* original_queryData);

// Returns true if the string represents a non-negative number, eitherwise returns false.
bool isNumber(char *word) {
    if (word == NULL) return false;
    // Check each character if it is digit.
    for (int i=0; i<(int) strlen(word); i++)
        if (!isdigit(word[i])) return false;
    return true;
}

// Executable should be called with: 
// ./search -d <input_file_original> -i <input_file_new> -s <query_file_new> -q <query_file_original> -ο <output_file> -k <int> -L <int> -Ν <number_of_nearest> -R <radius>
int main(int argc, char** argv) {

//------------------------------------------------------------------------------------
// Reading inline parameters.
    int i;

// Search for <-d> parameter.
	for (i=1; i < argc - 1; i++)
		if (strcmp(argv[i], "-d") == 0) break;
	if (i >= argc - 1) {
      	cout << "\033[0;31mError!\033[0m Not included '-d' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -q <query_file_original> -s <query_file_new> -ο <output_file> -k <int> -L <int>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
		return 1;
	}
	char *original_inputFile = argv[i+1];

// Search for <-i> parameter.
    for (i=1; i < argc - 1; i++)
        if (strcmp(argv[i], "-i") == 0) break;
    if (i >= argc - 1) {
        cout << "\033[0;31mError!\033[0m Not included '-i' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -q <query_file_original> -s <query_file_new> -ο <output_file> -k <int> -L <int>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
        return 1;
    }
    char *reduced_inputFile = argv[i+1];

// Search for <-q> parameter.
	for (i=1; i < argc - 1; i++)
		if (strcmp(argv[i], "-q") == 0) break;
	if (i >= argc - 1) {
      	cout << "\033[0;31mError!\033[0m Not included '-q' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -q <query_file_original> -s <query_file_new> -ο <output_file> -k <int> -L <int>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
		return 1;
	}
	char *original_queryFile = argv[i+1];

// Search for <-s> parameter.
    for (i=1; i < argc - 1; i++)
        if (strcmp(argv[i], "-s") == 0) break;
    if (i >= argc - 1) {
        cout << "\033[0;31mError!\033[0m Not included '-q' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -q <query_file_original> -s <query_file_new> -ο <output_file> -k <int> -L <int>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
        return 1;
    }
    char *reduced_queryFile = argv[i+1];

// Search for <-o> parameter.
	for (i=1; i < argc - 1; i++)
		if (strcmp(argv[i], "-o") == 0) break;
	if (i >= argc - 1) {
      	cout << "\033[0;31mError!\033[0m Not included '-o' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -q <query_file_original> -s <query_file_new> -ο <output_file> -k <int> -L <int>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
		return 1;
	}
	char *outputFile = argv[i+1];

// Search for <-k> parameter.
	for (i=1; i < argc - 1; i++)
		if (strcmp(argv[i], "-k") == 0) break;
    if (i >= argc - 1) {
        cout << "\033[0;31mError!\033[0m Not included '-k' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -q <query_file_original> -s <query_file_new> -ο <output_file> -k <int> -L <int>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
        return 1;
    } else if (!isNumber(argv[i+1])) {
    // <-k> parameter is invalid.
  	    cout << "\033[0;31mError!\033[0m Invalid value on '-k' parameter." << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
        return 1;        
    }
    int k = atoi(argv[i+1]);

// Search for <-L> parameter.
	for (i=1; i < argc - 1; i++)
		if (strcmp(argv[i], "-L") == 0) break;
    if (i >= argc - 1) {
        cout << "\033[0;31mError!\033[0m Not included '-L' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -q <query_file_original> -s <query_file_new> -ο <output_file> -k <int> -L <int>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
        return 1;
    } else if (!isNumber(argv[i+1])) {
    // <-L> parameter is invalid.
  	    cout << "\033[0;31mError!\033[0m Invalid value on '-L' parameter." << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
        return 1;        
    }
    int L = atoi(argv[i+1]);

// // Search for <-N> parameter.
    // int N = 1;
// 	for (i=1; i < argc - 1; i++)
// 		if (strcmp(argv[i], "-N") == 0) break;
// 	if (i < argc - 1) {
//       	if (!isNumber(argv[i+1])) {
//         // <-N> parameter is invalid.
//       	    cout << "\033[0;31mError!\033[0m Invalid value on '-N' parameter." << endl;
//             cout << "\033[0;31mExit program.\033[0m" << endl;
//             return 1;        
//         }
//         N = atoi(argv[i+1]);
//     }

// // Search for <-R> parameter.
    // double R = 1.0;
// 	for (i=1; i < argc - 1; i++)
// 		if (strcmp(argv[i], "-R") == 0) break;
// 	if (i < argc - 1) {
//       	if (!isNumber(argv[i+1])) {
//         // <-R> parameter is invalid.
//       	    cout << "\033[0;31mError!\033[0m Invalid value on '-R' parameter." << endl;
//             cout << "\033[0;31mExit program.\033[0m" << endl;
//             return 1;        
//         }
//         R = atoi(argv[i+1]);
//     }

//------------------------------------------------------------------------------------
// Reading input files.

    // Check that input file exists.
    if(access(original_inputFile, F_OK) == -1) {
        perror("\033[0;31mError\033[0m: Unable to open original input file");
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }
    cout << "\033[0;36mRunning LSH :)\033[0m" << endl << endl;
    // Read input file with PandaC.
    NumC<int>* original_inputData = PandaC<int>::fromMNIST(original_inputFile, 10000);

    // Check that input file exists.
    if(access(reduced_inputFile, F_OK) == -1) {
        perror("\033[0;31mError\033[0m: Unable to open reduced input file");
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }
    cout << "\033[0;36mRunning LSH :)\033[0m" << endl << endl;
    // Read input file with PandaC.
    NumC<int>* reduced_inputData = PandaC<int>::fromMNISTnew(reduced_inputFile, 10000);
    // reduced_inputData->print();

//------------------------------------------------------------------------------------
// Reading query files.

    // Check that input file exists.
    if(access(original_queryFile, F_OK) == -1) {
        perror("\033[0;31mError\033[0m: Unable to open original query file");
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }
    // Read query file with PandaC.
    NumC<int>* original_queryData = PandaC<int>::fromMNIST(original_queryFile, 10);

    // Check that input file exists.
    if(access(reduced_queryFile, F_OK) == -1) {
        perror("\033[0;31mError\033[0m: Unable to open reduced query file");
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }
    // Read query file with PandaC.
    NumC<int>* reduced_queryData = PandaC<int>::fromMNISTnew(reduced_queryFile, 10);

//------------------------------------------------------------------------------------
// Making predictions.

char line[128], *answer;
Results *lsh_results_original, *knn_results_reduced, *true_results;
vector<Results*> r_results;

    do {
        cout << "\033[0;36mComputing predictions...\033[0m" << endl << endl;
    //------------------------------------------------------------------------------------
    // Call LSHashing classifier and train it.

        LSHashing<int> lsh_original(L, k, 20000);
        lsh_original.fit_transform(original_inputData);

    //------------------------------------------------------------------------------------
    // Call exhaustive knn classifier and train it.

        ExhaustiveKnn<int> exhaustive_original(original_inputData, 1);
        ExhaustiveKnn<int> exhaustive_reduced(reduced_inputData, 1);

    //------------------------------------------------------------------------------------
    // Execute Predictions and extract results to output file.

        // Execute k-NN prediction.
        lsh_results_original = lsh_original.predict_knn(original_queryData, 1);
        knn_results_reduced = exhaustive_reduced.predict_knn(reduced_queryData, 1);
        // Execute Exhaustive KNN search.
        true_results = exhaustive_original.predict_knn(original_queryData, 1);
        
        // // Execute Range Search.
        // r_results = lsh.predict_rs(original_queryData, R);

        // // Extract results on output file.
        if (extractResults(outputFile, lsh_results_original, knn_results_reduced, true_results, original_inputData, original_queryData) ) {
            cout << "Results are extracted in file: " << outputFile << endl;
        }

    //------------------------------------------------------------------------------------
    // Free allocated Space.

        delete original_queryData;
        delete reduced_queryData;
        delete lsh_results_original;
        delete knn_results_reduced;
        delete true_results;
        for (int i=0; i < (int) r_results.size(); i++) {
            delete r_results[i];
        }

    //------------------------------------------------------------------------------------
    // Ask user if he wants to repeat the process with new query file.

        cout << "-----------------------------------------------------------------" << endl;
    //     do {
    //         cout << "\033[0;36mYou would like to repeat the process with new query? (answer y|n) \033[0m";
    // 		fgets(line,sizeof(line),stdin);
    // 		answer = strtok(line,"\n");
    //     } while (strcmp(answer, "n") && strcmp(answer, "N") && strcmp(answer, "y") && strcmp(answer, "Y"));
    //     if (!strcmp(answer, "y") || !strcmp(answer, "Y")) {
    //     // User wants to repeat the process.
    //         cout << "\033[0;36mPlease enter a new query file (press Enter to use the old one): \033[0m";
    // 		fgets(line,sizeof(line),stdin);
    //         if (strlen(line) > 1) {
    //             queryFile = strtok(line,"\n");
    //             // Check that query file exists.
    //             if(access(queryFile, F_OK) == -1) {
    //                 perror("\033[0;31mError\033[0m: Unable to open query file");
    //                 cout << "\033[0;31mexit program\033[0m" << endl;
    //                 return 1;
    //             }
    //             // Read query file with PandaC.
    //             queryData = PandaC<int>::fromMNIST(queryFile);
    //         } else {
    //             if(access(queryFile, F_OK) == -1) {
    //                 perror("\033[0;31mError\033[0m: Unable to open query file");
    //                 cout << "\033[0;31mexit program\033[0m" << endl;
    //                 return 1;
    //             }
    //             // Read query file with PandaC.
    //             queryData = PandaC<int>::fromMNIST(queryFile);
    //         }
    //         cout << "\033[0;36mPlease enter an output file (press Enter to use the old one): \033[0m";
    // 		fgets(line,sizeof(line),stdin);
    //         cout << endl;
    //         if (strlen(line) > 1) {
    //     		outputFile = strtok(line,"\n");
    //         }
    //     }
    // } while (strcmp(answer, "n") && strcmp(answer, "N"));
    } while (false);

//------------------------------------------------------------------------------------
// End of program.

    delete original_inputData;
    delete reduced_inputData;

    // cout << endl << "-----------------------------------------------------------------" << endl;
    cout << "\033[0;36mExit program.\033[0m" << endl;
    return 0;
}

bool extractResults(char* outputFile, Results* results_original, Results* results_reduced, Results *true_results, NumC<int>* original_inputData, NumC<int>* original_queryData) {
    // NumC<int>* inputDatalabels = PandaC<int>::fromMNISTlabels((char*) "./doc/input/train-labels-idx1-ubyte");

    // Check that output file exists.
    ofstream output(outputFile, fstream::out);
    if (!output.is_open()) {
        perror("\033[0;31mError\033[0m: Unable to open output file");
        return false;
    }

    double sumDistanceReducedRatio = 0;
    double sumTimeReduced = 0;
    double sumDistanceLSHRatio = 0;
    double sumTimeLSH = 0;
    double sumDistanceTrue = 0;
    double sumTimeTrue = 0;
    NumCDistType originalDistance = 0;
    for (int i=0; i < results_original->resultsIndexArray.getRows(); i++) {
        output << "Query: " << i+1 << endl;

        output << "  Nearest neighbor Reduced: " << results_reduced->resultsIndexArray.getElement(i, 0) << endl;
        output << "  Nearest neighbor LSH: " << results_original->resultsIndexArray.getElement(i, 0) << endl;
        output << "  Nearest neighbor True: " << true_results->resultsIndexArray.getElement(i, 0) << endl;
        // output << "  Nearest neighbor-" << j+1 << ": " << results->resultsIndexArray.getElement(i, j) << endl;
        // // output << "  Nearest neighbor-" << j+1 << ": " << results->resultsIndexArray.getElement(i, j) << " label: " << inputDatalabels->getElement(results->resultsIndexArray.getElement(i, j) ,0) << " true label: " << inputDatalabels->getElement(true_results->resultsIndexArray.getElement(i, j) ,0)<<endl;
        
        output << "   distanceReduced:  " << results_reduced->resultsDistArray.getElement(i, 0) << endl;
        originalDistance = NumC<int>::dist(original_inputData->getVector(results_reduced->resultsIndexArray.getElement(i,0)), original_queryData->getVector(i), 1);
        sumDistanceReducedRatio += (originalDistance / true_results->resultsDistArray.getElement(i, 0));
        output << "   distanceLSH:  " << results_original->resultsDistArray.getElement(i, 0) << endl;
        sumDistanceLSHRatio += (results_original->resultsDistArray.getElement(i, 0) / true_results->resultsDistArray.getElement(i, 0));
        output << "   distanceTrue: " << true_results->resultsDistArray.getElement(i, 0) << endl;
        sumDistanceTrue += true_results->resultsDistArray.getElement(i, 0);
        
        
        sumTimeReduced += results_reduced->executionTimeArray.getElement(i, 0);
        sumTimeLSH += results_original->executionTimeArray.getElement(i, 0);
        sumTimeTrue += true_results->executionTimeArray.getElement(i, 0);
        
    }

    output<<endl;
    output << "tReduced: " << sumTimeReduced << endl;
    output << "tLSH: " << sumTimeLSH << endl;
    output << "tTrue: " << sumTimeTrue << endl;
    output << "Approximation Factor LSH: " << sumDistanceLSHRatio / results_original->resultsIndexArray.getRows() << endl;
    output << "Approximation Factor Reduced: " << sumDistanceReducedRatio / results_original->resultsIndexArray.getRows() << endl;
    output<<endl;

    cout<<endl;
    cout << "tReduced: " << sumTimeReduced << endl;
    cout << "tLSH: " << sumTimeLSH << endl;
    cout << "tTrue: " << sumTimeTrue << endl;
    cout << "Approximation Factor LSH: " << sumDistanceLSHRatio / results_original->resultsIndexArray.getRows() << endl;
    cout << "Approximation Factor Reduced: " << sumDistanceReducedRatio / results_original->resultsIndexArray.getRows() << endl;
    cout<<endl;

    // Printing Score of prediction.
    cout << "Final Exhaustive vs LSH Score:" << endl;

    // cout << "Average True distance on reduced dataset: " << (double) sumDistanceReducedRatio/(1*results_original->resultsIndexArray.getRows()) << endl;
    // cout << "Average LSH distance on original dataset: " << (double) sumDistanceLSHRatio/(1*results_original->resultsIndexArray.getRows()) << endl;
    // cout << "Average True distance: " << (double) sumDistanceTrue/(1*results_original->resultsIndexArray.getRows()) << endl;
    
    cout << "True(Reduced)/True distance: " << (double) sumDistanceReducedRatio / results_original->resultsIndexArray.getRows() << " (LSH has better accuracy when this number is inclined to 1)" << endl;
    cout << "LSH(Original)/True distance: " << (double) sumDistanceLSHRatio / results_original->resultsIndexArray.getRows() << " (LSH has better accuracy when this number is inclined to 1)" << endl;
    
    cout << "True(Reduced)'s fault is " << (int) ((((double) sumDistanceReducedRatio/ results_original->resultsIndexArray.getRows() )-1)*100) << "% on average prediction." << endl;
    cout << "LSH(Original)'s fault is " << (int) ((((double) sumDistanceLSHRatio / results_original->resultsIndexArray.getRows() )-1)*100) << "% on average prediction." << endl;
    cout << endl;
    
    cout << "Average Time on reduced dataset: " << (double) sumTimeReduced/results_reduced->resultsIndexArray.getRows() << endl;
    cout << "Average True Time on original dataset: " << (double) sumTimeTrue/results_original->resultsIndexArray.getRows() << endl;
    
    cout << "True(Reduced)/True Time: " << (double) sumTimeReduced/sumTimeTrue << " (LSH is faster when this number is inclined to 0)" << endl;
    cout << "LSH(Original)/True Time: " << (double) sumTimeLSH/sumTimeTrue << " (LSH is faster when this number is inclined to 0)" << endl;
    
    if (sumTimeLSH > sumTimeTrue) {
        cout << "True(Reduced)'s time is " << (int) ((((double) sumTimeReduced/sumTimeTrue)-1)*100) << "% slower than true." << endl;
        cout << "LSH(Original)'s time is " << (int) ((((double) sumTimeLSH/sumTimeTrue)-1)*100) << "% slower than true." << endl;
    } else {
        cout << "True(Reduced)'s time is " << (int) ((1-((double) sumTimeReduced/sumTimeTrue))*100) << "% faster than true." << endl;
        cout << "LSH(Original)'s time is " << (int) ((1-((double) sumTimeLSH/sumTimeTrue))*100) << "% faster than true." << endl;
    }
    cout << endl;

    // Close output file.
    output.close();
    // delete inputDatalabels;

    return true;
}