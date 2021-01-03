import numpy as np
import os
import sys
import struct
import json
from array import array as pyarray
import matplotlib.pyplot as plt
import time


# Define class with colors for UI improvement
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def load_mnist(dataset, digits=np.arange(10), type='data', numOfElements=-1):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    if not os.path.isfile(dataset):
        return None
    fname = os.path.join(".", dataset)
    if (type == 'data'):
        nMetaDataBytes = 4 * intType.itemsize
        images = np.fromfile(fname, dtype = 'ubyte')
        magicBytes, size, rows, cols = np.frombuffer(images[:nMetaDataBytes].tobytes(), intType)
        if numOfElements == -1:
            numOfElements = size #int(len(ind) * size/100.)
        images = images[nMetaDataBytes:].astype(dtype = 'float32').reshape([numOfElements, rows, cols, 1])
        return images
    elif (type == 'labels'):
        nMetaDataBytes = 2 * intType.itemsize
        labels = np.fromfile(fname, dtype = 'ubyte')[nMetaDataBytes:]
        return labels
    else:
        return None


#  distance
def manhattan_distances(data, queries):
    
    # empty matrix for resutls
    results = np.empty((queries.shape[0], data.shape[0]), dtype=np.float32)

    # fill the manhattan distance of every query
    for query in range(queries.shape[0]):
        l1_norm = np.linalg.norm((data - queries[query]), ord=1, axis=1)  
        results[query] = l1_norm 

    return results.T

# manhattan distance
def euklidian_distances(data, queries):
    
    # empty matrix for resutls
    results = np.empty((queries.shape[0], data.shape[0]), dtype=np.float32)

    # fill the manhattan distance of every query
    for query in range(queries.shape[0]):
        l1_norm = np.linalg.norm((data - queries[query]), axis=1)  
        results[query] = l1_norm 

    return results.T


def earths_movers_distances(data, queries, distances, A):
    
    # empty matrix for resutls
    results = np.empty((len(queries), len(data)), dtype=np.float32)

    # fill the manhattan distance of every query
    for query in range(len(queries)):
        for data_i in range(len(data)):
            emd = earths_movers_distance(data[data_i], queries[query], distances=distances, A=A)
            # emd = ot.emd2(data[data_i], queries[query],distances_array)
            results[query][data_i] = emd

    return results.T


# KNN classifier
class KNN():

    def __init__(self, n_neighbors=10, distance_function=manhattan_distances, distances=None, A=None):
        self.distance_function = distance_function
        self.n_neighbors = n_neighbors
        self.prediction_time = 0
        # emd distances
        self.distances = distances
        self.A = A 

    def fit(self, x_train, y_train):
        # place self_x in an array
        self.x_train = x_train
        self.y_train = y_train

        # regcognise different classes
        self.classes = list(np.unique(y_train))
        self.classes.sort()

    def predict(self, x_test):
        # strt the prdiciton timer
        start_time = time.time()

        # get distances for all the vectors
        if self.distances is None:
            distances = self.distance_function(self.x_train, x_test)
        else: # earths movers distance
            distances = self.distance_function(self.x_train, x_test, self.distances, self.A)

        self.y_pred = self.y_train[np.argpartition(distances.T, self.n_neighbors)][:,0:self.n_neighbors]

        # get the prediction time
        end_time = time.time()
        self.prediction_time = end_time - start_time

        return self.y_pred.reshape((-1, self.n_neighbors))
    
    def predict_proba(self, x_test):
        self.predict(x_test)
        # predict propabilities
        propabilities = []
        for prediction in self.y_pred.tolist():
            propability = [prediction.count(cl)/self.n_neighbors for cl in self.classes]
            propabilities.append(propability)
        
        return np.array(propabilities)

def get_Accuracy(y_pred, y_true):
    
    results = y_pred - y_true

    # zeros are the correct so calculate them
    mean_accuracies = np.mean(results == 0, axis=1)
    # get the mean accuracy
    mean_accuracy   = np.mean(mean_accuracies)
    return mean_accuracy


###########################################################################
################################### EMD ###################################
###########################################################################


# manipulate strides to create a winowed resize of the 2d array
# ideas were taken from https://github.com/scikit-image/scikit-image/blob/master/skimage/util/shape.py
from numpy.lib.stride_tricks import as_strided
def get_Windowed_view(array_in, shape):

    windows_in_image = np.array(array_in.shape) / np.array(shape)
    new_shape = list(windows_in_image.astype(int)) + shape
    new_strides = (np.array(array_in.strides)*shape[0]).tolist() + list(array_in.strides)
    arr_out = as_strided(array_in, shape=new_shape, strides=new_strides)
    return arr_out

# get the clusters as a list from all the images
def get_Clusters(images, window):

    # normalize images
    images = images.reshape(images.shape[0:-1])
    # images = images / images.sum(axis=(1,2), keepdims=True) 

    cluster_weights = []
    for image in images:
        # get the windowed image
        windowed_image = get_Windowed_view(image, window)
        # sum the windows-clusters
        windowed_image = np.sum(windowed_image, axis=(2,3)) 
        # cluster_weights.append( windowed_image.tolist() )
        cluster_weights.append( windowed_image.reshape((-1)).tolist() )

    return cluster_weights


# get the euklidian diastnces of the clusters
def get_Clusters_distances(dim, image_shape):
    clusters = []


    for i in range(0, image_shape[0], dim):
        for j in range(0, image_shape[1], dim):
            cluster = [i,j]
            clusters.append(cluster)
    
    clusters = np.array(clusters)

    # euklidian distances n*n matrix
    distances = euklidian_distances(clusters,clusters)
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    distances_array = distances

    # reshape and make it a list for the scipy linprog
    distances = distances.reshape((-1)).tolist()
    return distances, distances_array

def get_A(num_of_weights, num_of_variables):
    """
    A matrix should be like:

    Α = [[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
        [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
        [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
        [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]]

    in order to satisfy the constraints of the EMD distance
    """
    A = np.zeros((num_of_weights, num_of_variables))    

    # fill first half
    mask_ones = int(num_of_weights/2)
    mask = np.ones((1, mask_ones))
    for i in range(mask_ones):
        A[i][i*mask_ones : i*mask_ones + mask_ones] = mask 

    # second half
    one_offset = 0
    for i in range(mask_ones, num_of_weights, 1):
        for j in range(mask_ones):
            A[i][j*mask_ones + one_offset] = 1
        one_offset += 1
    
    return A

# implement EMD as proposed in
# https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf
# linear programming
from scipy.optimize import linprog
def earths_movers_distance(image1_clustes, image2_clustes, distances, A):

    # distances dij
    c = distances

    # weights
    b = image1_clustes + image2_clustes

    # linear optimization
    # res = linprog(c, A_eq=A, b_eq=b, method='revised simplex', options={"tol":1e-7})
    # res = linprog(c, A_eq=A, b_eq=b, options={"tol":1e-5})

    Aeq = np.ones((1, A.shape[1]))
    beq = min(sum(image1_clustes), sum(image2_clustes))
    res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, method='revised simplex')

    return res.fun / sum(res.x)






if __name__ == '__main__':

    # Reading inline arguments
    # <-d> argument
    if '-d' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-d>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-d') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
            sys.exit()
        datasetFile = sys.argv[sys.argv.index('-d')+1]
    # <-dl> argument
    if '-q' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-q>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-q') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
            sys.exit()
        testsetFile = sys.argv[sys.argv.index('-q')+1]
    # <-t> argument
    if '-l1' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-l1>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-l1') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
            sys.exit()
        dlabelsFile = sys.argv[sys.argv.index('-l1')+1]
    # <-tl> argument
    if '-l2' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-l2>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-l2') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
            sys.exit()
        tlabelsFile = sys.argv[sys.argv.index('-l2')+1]
    # <-model> argument
    if '-o' not in sys.argv:
        print(bcolors.FAIL+'Error: missing argument <-o>.'+bcolors.ENDC)
        print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
        sys.exit()
    else:
        if sys.argv.index('-o') == len(sys.argv)-1:
            print(bcolors.FAIL+'Error: invalid arguments.'+bcolors.ENDC)
            print(bcolors.WARNING+'Executable should be called:', sys.argv[0], '–d  <input  file  original  space>  –q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file>'+bcolors.ENDC)
            sys.exit()
        outputFile = sys.argv[sys.argv.index('-o')+1]


    # datasetFile = "/content/Latent_vs_Original_Space_Image_Classification/data/train-images-idx3-ubyte"
    # dlabelsFile = "/content/Latent_vs_Original_Space_Image_Classification/data/train-labels-idx1-ubyte"
    # testsetFile = "/content/Latent_vs_Original_Space_Image_Classification/data/t10k-images-idx3-ubyte"
    # tlabelsFile = "/content/Latent_vs_Original_Space_Image_Classification/data/t10k-labels-idx1-ubyte"
    # print(datasetFile,dlabelsFile,testsetFile,tlabelsFile, outputFile)

    t = 100
    q = 1

    train_X = load_mnist(datasetFile, type='data')[0:t]
    train_Y = load_mnist(dlabelsFile, type='labels')[0:t]
    test_X  = load_mnist(testsetFile, type='data')[0:q]
    test_Y  = load_mnist(tlabelsFile, type='labels')[0:q]

    # reshape labels
    train_Y = train_Y.reshape((-1,1))
    test_Y  = test_Y.reshape((-1,1))

    input_shape = train_X.shape[1:]
    num_of_classes = train_Y.shape


    ###########################################################################
    ################################### KNN ###################################
    ###########################################################################


    # preprocess for knn manhattan
    x_train = train_X.reshape((train_X.shape[0], -1))
    x_test = test_X.reshape((test_X.shape[0], -1))

    # init knn classifier
    manhattan_knn = KNN(10, manhattan_distances)
    manhattan_knn.fit(x_train, train_Y)
    manhattan_predictions = manhattan_knn.predict(x_test)

    manhattan_accuracy = get_Accuracy(manhattan_predictions, test_Y)

    # print accuracy and time
    manhattan_message = "Average Correct Search Results MANHATTAN: %0.4f in Time: %0.5f seconds" % (manhattan_accuracy, manhattan_knn.prediction_time)
    print(manhattan_message)



    
    ###########################################################################
    ################################### EMD ###################################
    ###########################################################################


    # preprocess for EMD
    dim = 14

    # get clusters of train and test
    train_clusters = get_Clusters(train_X, [dim, dim])
    test_clusters = get_Clusters(test_X, [dim, dim])

    # get distances
    distances, distances_array = get_Clusters_distances(dim, input_shape[:-1])

    # get the number of variables
    # 2 times the weights for the linprog
    num_of_weights = 2*len(train_clusters[0])
    # variables are the of distances
    num_of_variables = len(distances)
    # get A for the EMD coefficients
    A = get_A(num_of_weights, num_of_variables)

    # emd_knn
    emd_knn = KNN(10, earths_movers_distances, distances, A)
    emd_knn.fit(train_clusters, train_Y)
    emd_predictions = emd_knn.predict(test_clusters)

    emd_accuracy = get_Accuracy(emd_predictions, test_Y)

    # print accuracy and time
    emd_message = "Average Correct Search Results EMD:       %0.4f in Time: %0.5f seconds with cluster size: %d" % (emd_accuracy, emd_knn.prediction_time, dim)
    print(emd_message)


    with open(outputFile, "w") as f: 
        f.write(emd_message+"\n")
        f.write(manhattan_message+"\n")