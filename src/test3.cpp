// #include <iostream>
// #include <string.h>
// #include <unistd.h>
// #include <vector>
// #include <fstream>

// // #include "../include/pandac.h"
// // #include "../include/kmedians.h"
// #include "../include/numc.h"
// // #include "../include/prediction_results.h"
// // #include "../include/exhaustive_knn.h"
// // #include "../include/lsh_classifier.h"


// using namespace std;


// NumC<int>* readClusters(char* clustersFile, int inputDataSize) {
//     NumC<int> *clusters = new NumC<int>(inputDataSize, 1);
//     FILE *cFile = fopen(clustersFile, "r");
    
//     char line[128];
//     // char *command;
//     // int value;
//     int index;
//     int clusterSize = 0;
//     bool newLine = true;
//     int clusterNum = 0;
//     while (!feof(cFile)) {
//     // Read "CLUSTER-<i> { size: <int>,"
//         fscanf(cFile, "CLUSTER-%d { size: %d,", &index, &clusterSize);
//         cout << index << " - " << clusterSize << endl;
//         if (index != clusterNum+1 && clusterSize > inputDataSize) return NULL;
//     // Read " <int>", custerSize times
//         for (int j=0; j<clusterSize; j++) {
//             fscanf(cFile, " %d,", &index);
//             if (j<5) cout << index << endl;
//             if (index < 0) return NULL;
//             clusters->addElement(clusterNum, index, 0);
//         }
//     // Read "}\n"
//         fscanf(cFile, " } ");
//         clusterNum++;
//     }
//     // while(fgets(line,sizeof(line),c) != NULL) {
//         // CLUSTER-1 { size: <int>, image_numberA, ..., image_numberX}

//         // if (strlen(line) > 1) {
//         //     // cout << line;
//         //     command = strtok(line," : ");
//         //     value = atoi(strtok(NULL,"\n"));
//         // }
//         // // cout << "<" << command << ">: " << value << endl;
//         // if (!strcmp(command, (char*) "number_of_clusters")) {
//         //     confData.number_of_clusters = value;
//         // } else if (!strcmp(command, (char*) "number_of_vector_hash_tables")) {
//         //     confData.L = value;
//         // } else if (!strcmp(command, (char*) "number_of_vector_hash_functions")) {
//         //     confData.k = value;
//         // } else if (!strcmp(command, (char*) "max_number_M_hypercube")) {
//         //     confData.M = value;
//         // } else if (!strcmp(command, (char*) "number_of_hypercube_dimensions")) {
//         //     confData.d = value;
//         // } else if (!strcmp(command, (char*) "number_of_probes")) {
//         //     confData.probes = value;
//         // } else {
//         //     // Not accepted configuration
//         // }
//     //     clusters[i] = clusterNum;
//     //     clusterNum++;
//     // }

//     fclose(cFile);
//     return clusters;
// }

// int main(int argc, char** argv) {
//     // NumC<int>* queryData = PandaC<int>::fromMNISTnew(argv[1], 10);
//     // cout << "+++++++++++++++++++++++++++++" << endl;
//     // cout << "END OF MINST" << endl;
//     // cout << "+++++++++++++++++++++++++++++" << endl;
//     // queryData->print();


//     NumC<int>* clusters = readClusters(argv[1], 20);
//     cout << "+++++++++++++++++++++++++++++" << endl;
//     cout << "END OF <readClusters>" << endl;
//     cout << "+++++++++++++++++++++++++++++" << endl;
//     clusters->print();
//     return 0;
// }