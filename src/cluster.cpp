#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <fstream>

#include "../include/pandac.h"
#include "../include/numc.h"
#include "../include/prediction_results.h"
#include "../include/kmedians.h"
#include "../include/exhaustive_knn.h"
#include "../include/lsh_classifier.h"

using namespace std;

// Extracts the results to output file.
template <typename NumCDataType> 
bool extractResults(char* outputFile, char* method, bool complete, Kmedians<NumCDataType> *kMedians);

// Returns true if the string represents a non-negative number, eitherwise returns false.
bool isNumber(char *word) {
    if (word == NULL) return false;
    // Check each character if it is digit.
    for (int i=0; i<(int) strlen(word); i++)
        if (!isdigit(word[i])) return false;
    return true;
}

// Executable should be called with: 
// ./cluster –i <input_file> –c <configuration_file> -o <output_file> -m <method: 'Classic' OR 'LSH' OR 'Hypercube'> -complete (optional)
int main(int argc, char** argv) {

//------------------------------------------------------------------------------------
// Reading inline parameters.
    int i;

// Search for <-d> parameter.
    for (i=1; i < argc - 1; i++)
        if (strcmp(argv[i], "-d") == 0) break;
    if (i >= argc - 1) {
        cout << "\033[0;31mError!\033[0m Not included '-d' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -n <clusters_file> -c <configuration_file> -o <output_file>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
        return 1;
    }
    char *original_inputFile = argv[i+1];

// Search for <-i> parameter.
	for (i=1; i < argc - 1; i++)
		if (strcmp(argv[i], "-i") == 0) break;
	if (i >= argc - 1) {
      	cout << "\033[0;31mError!\033[0m Not included '-i' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -n <clusters_file> -c <configuration_file> -o <output_file>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
		return 1;
	}
	char *reduced_inputFile = argv[i+1];	

// Search for <-c> parameter.
	for (i=1; i < argc - 1; i++)
		if (strcmp(argv[i], "-c") == 0) break;
	if (i >= argc - 1) {
      	cout << "\033[0;31mError!\033[0m Not included '-c' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -n <clusters_file> -c <configuration_file> -o <output_file>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
		return 1;
	}
	char *configurationFile = argv[i+1];	

// Search for <-n> parameter.
    for (i=1; i < argc - 1; i++)
        if (strcmp(argv[i], "-n") == 0) break;
    if (i >= argc - 1) {
        cout << "\033[0;31mError!\033[0m Not included '-n' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -n <clusters_file> -c <configuration_file> -o <output_file>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
        return 1;
    }
    char *clustersFile = argv[i+1];

// Search for <-o> parameter.
	for (i=1; i < argc - 1; i++)
		if (strcmp(argv[i], "-o") == 0) break;
	if (i >= argc - 1) {
      	cout << "\033[0;31mError!\033[0m Not included '-o' parameter." << endl;
        cout << "Executable should be called with: " << argv[0] << " -d <input_file_original> -i <input_file_new> -n <clusters_file> -c <configuration_file> -o <output_file>" << endl;
        cout << "\033[0;31mExit program.\033[0m" << endl;
		return 1;
	}
	char *outputFile = argv[i+1];

// // Search for <-m> parameter.
// 	for (i=1; i < argc - 1; i++)
// 		if (strcmp(argv[i], "-m") == 0) break;
// 	if (i < argc - 1) {
//         if (i >= argc - 1) {
//             cout << "\033[0;31mError!\033[0m Not included '-m' parameter." << endl;
//             cout << "Executable should be called with: " << argv[0] << " –i <input_file> –c <configuration_file> -o <output_file> -m <method: 'Classic' OR 'LSH' OR 'Hypercube'> -complete (optional)" << endl;
//             cout << "\033[0;31mExit program.\033[0m" << endl;
//             return 1;
//         }
// 	}
//     if (!strcmp(argv[i+1], (char*) "Classic") && !strcmp(argv[i+1], (char*) "LSH") && !strcmp(argv[i+1], (char*) "Hypercube")) {
//         cout << "\033[0;31mError!\033[0m Invalid method.\n" << endl;
//         cout << "\033[0;31mExit program.\033[0m" << endl;
//         return 1;
//     }
// 	char *method = argv[i+1];

// // Search for <-complete> parameter.
//     bool complete = false;
// 	for (i=1; i < argc; i++)
// 		if (strcmp(argv[i], "-complete") == 0) break;
// 	if (i < argc) {
//         complete = true;
//     }

//------------------------------------------------------------------------------------
// Reading input files.

    // Check that input file exists.
    if(access(original_inputFile, F_OK) == -1) {
        perror("\033[0;31mError\033[0m: Unable to open the original input file");
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }
    cout << "\033[0;36mRunning Cluster :)\033[0m" << endl << endl;
    // Read input file with PandaC.
    NumC<int>* original_inputData = PandaC<int>::fromMNIST(original_inputFile, 2000);

    // Check that input file exists.
    if(access(reduced_inputFile, F_OK) == -1) {
        perror("\033[0;31mError\033[0m: Unable to open the new input file");
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }
    cout << "\033[0;36mRunning Cluster :)\033[0m" << endl << endl;
    // Read input file with PandaC.
    NumC<int>* reduced_inputData = PandaC<int>::fromMNISTnew(reduced_inputFile, 2000);

//------------------------------------------------------------------------------------
// Reading configuration file.

    // Check that configurationn file exists.
    if(access(configurationFile, F_OK) == -1) {
        perror("\033[0;31mError\033[0m: Unable to open the configurationn file");
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }
    // Read configuration file.
    ConfigurationData conf = readConfiguration(configurationFile);
    if (conf.isEmpty()) {
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }

//------------------------------------------------------------------------------------
// Reading clusters from clustersFIle.

    // Check that configurationn file exists.
    if(access(clustersFile, F_OK) == -1) {
        perror("\033[0;31mError\033[0m: Unable to open the clusters file");
        cout << "\033[0;31mexit program\033[0m" << endl;
        return 1;
    }
    // Read configuration file.
    NumC<int>* clusters = readClusters(clustersFile, original_inputData->getRows());
    


//------------------------------------------------------------------------------------
// Making predictions.

    cout << "\033[0;36mComputing clusters...\033[0m" << endl << endl;

//------------------------------------------------------------------------------------
// Call K-Medians and train it.

    // Kmedians<int> kMedians(conf);
    // if (!strcmp(method, (char*) "Classic")) {
    //     kMedians.fit_transform(inputData, LLOYDS_CLUSTERING);
    // } else if (!strcmp(method, (char*) "LSH")) {
    //     kMedians.fit_transform(inputData, LSH_CLUSTERING);
    // } else if (!strcmp(method, (char*) "Hypercube")) {
    //     kMedians.fit_transform(inputData, HC_CLUSTERING);
    // }

    Kmedians<int> original_space(conf);
    Kmedians<int> new_space(conf);
    Kmedians<int> classes_as_clusters(conf);

    cout <<endl<< "\033[0;36mNEW SPACE\033[0m" << endl;
    new_space.fit_transform(reduced_inputData, LLOYDS_CLUSTERING);
    // get clusters and start from all over again to compute in the original space
    NumC<int>* new_space_clusters = new NumC<int>(reduced_inputData->getRows(), 1, false);
    new_space.getLastResultsClusters(new_space_clusters);
    Kmedians<int> new_space_to_original(conf);
    new_space_to_original.fit(original_inputData);
    new_space_to_original.fit_clusters(new_space_clusters);
    cout << new_space_to_original.getObjectiveCost() <<endl;
    vector<NumCDistType> silhouette_new_space = new_space_to_original.getSilhouettes();
    // for (int i=0; i < 10; i++) {
    //     cout << silhouette_new_space[i] << ", ";
    // }
    // cout << silhouette_new_space[10] << "]" << endl; //!+++



    cout <<endl<< "\033[0;36mORIGINAL SPACE\033[0m" << endl;
    original_space.fit_transform(original_inputData, LLOYDS_CLUSTERING);
    cout << original_space.getObjectiveCost() <<endl;
    


    
    cout <<endl<< "\033[0;36mCLASSES AS CLUSTERS\033[0m" << endl;
    classes_as_clusters.fit(original_inputData);
    classes_as_clusters.fit_clusters(clusters);
    cout << classes_as_clusters.getObjectiveCost() <<endl;
    vector<NumCDistType> silhouette_from_clusters = classes_as_clusters.getSilhouettes();
    // for (int i=0; i < 10; i++) {
    //     cout << silhouette_from_clusters[i] << ", ";
    // }
    // cout << silhouette_from_clusters[10] << "]" << endl; //!+++


//------------------------------------------------------------------------------------
// Execute Predictions and extract results to output file.

    if (extractResults(outputFile, "Classic", true, &original_space)) {
        cout << "\033[0;36mResults are extracted in file: \033[0m" << outputFile << endl;
    }

//------------------------------------------------------------------------------------
// End of program.

    //Free allocated Space.
    delete original_inputData;
    delete reduced_inputData;
    delete clusters;
    delete new_space_clusters;

    cout << "-----------------------------------------------------------------" << endl;
    cout << "\033[0;36mExit program.\033[0m" << endl;
    return 0;
}

template <typename NumCDataType> 
bool extractResults(char* outputFile, char* method, bool complete, Kmedians<NumCDataType> *kMedians) {

    // Check that output file exists.
    ofstream output(outputFile, fstream::out);
    if (!output.is_open()) {
        perror("\033[0;31mError\033[0m: Unable to open output file");
        return false;
    }

    output << "Algorithm: ";
    if (!strcmp(method, (char*) "Classic")) {
        output << "Lloyds" << endl;
    } else if (!strcmp(method, (char*) "LSH")) {
        output << "Range Search LSH" << endl;
    } else if (!strcmp(method, (char*) "Hypercube")) {
        output << "Range Search Hypercube" << endl;
    }

    NumC<NumCDataType>* centroids = kMedians->getCentroids();
    vector<Results*> clusters = kMedians->getResults();
    vector<NumCDistType> silhouette = kMedians->getSilhouettes();
    for (int i=0; i < centroids->getRows(); i++) {
        output << "  CLUSTER-" << i+1 << " {size: " << clusters[i]->resultsIndexArray.getCols() << ", centroid: ";
        NumC<NumCDataType>::print(centroids->getVector(i), output);
        output << "}" << endl; //!+++
    }
    cout << endl;
    output << "  clustering_time: " << clusters[0]->executionTime << endl; //!+++
    output << "  Silhouette: [ ";
    cout << endl;
    for (int i=0; i < centroids->getRows(); i++) {
        output << silhouette[i] << ", ";
    }
    output << silhouette[centroids->getRows()] << "]" << endl; //!+++
    if (complete == true) {
        for (int i=0; i < centroids->getRows(); i++) {
            output << "  CLUSTER-" << i+1 << " {centroid: ";
            NumC<NumCDataType>::print(centroids->getVector(i), output);
            output << ", "; //!+++
            for (int j=0; j < clusters[i]->resultsIndexArray.getCols(); j++) {
                output << clusters[i]->resultsIndexArray.getElement(0, j);
                if (j+1 < clusters[i]->resultsIndexArray.getCols()) output << ", "; //!+++
            }
            output << "}" << endl;
        }
    }

    // Close output file.
    output.close();
    // Free allocated Space.
    // delete centroids;
    for (int i=0; i < (int) clusters.size(); i++) {
        delete clusters[i];
    }

    return true;
}
