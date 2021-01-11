<p style="text-align: center;">
    <img src="./docs/di_uoa.png" alt="UOA">
    <h1>University of Athens</h1>
    <h2>Department of Informatics and Telecomunications</h2>
</p>

<h3>Dionysis Taxiarchis Balaskas - 1115201700094</h3>
<h3>Andreas Giannoutsos - 1115201700021</h3>
<br>

# Latent vs Original Space Image Classification

<h3>Introduction to our project (info, goals, complexity, speed, results, simplicity, abstractiveness)</h3>
<h4>
Comparison of latent space produced from convolutional neuran nets vs original space af anImage classification and unsupervised learning using latent space vectors produced by convolutional neural nets together with the original vectors space image in classification and unsupervided clustering
</h4>

<h3>How we run the executables</h3>
<h4>
  To run it with google Colab:
</h4>

   [![Click here to open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AGiannoutsos/Image_Classification_with_Convolutional_Autoencoder/blob/main/experiments.ipynb)

<h3>How we tested that it works</h3>
 <h4>
    We tested our scripts and executables with every possible combination, based on the project requests. For better performance we used google colab for python scripts which provides powerfull hardware with memory and GPU. Google Colab minimized the time that each experiment took. We tested many hyperparameters' combinations and found some very good models with good performance and small loss. These two were the creterions which we used to consider which experiment was the best.
 </h4> 

<h3>Project Directories and Files organization</h3>
<h4>
  
  Main Directory:
    configurations/<br>       # Directory with hyperparameters configurations, saved on JSON form.<br><br>
    data/<br>                 # Directory with data files.<br><br>
    models/<br>               # Directory where models are saved.<br><br>
    outputs/<br>              # contains data files like input/query/output/configuration.<br><br>
    src/<br>                  # contains the '.cpp' files.<br><br>
    include/<br>              # contains the '.h' files.<br><br>
    experiments/<br>          # Directory where experiment's Python notebooks are saved.<br><br>
    emd/<br>                  # Directory where EMD implementation is.<br><br>
    docs/<br>                 # Directory with some documents and images from experiment.<br><br>
    bin/<br>                  # the directory where the C++ executables are saved.<br><br>
    build/<br>                # the directory where the C++ objective files are.<br><br>
    autoencoder.py<br>        # Autoencoder script.<br><br>
    reduce.py<br>             # reduce script.<br><br>
    classification.py<br>     # Classifier script.<br><br>
    model.py<br>              # Contains functions that are used for the Neural Network creation, train and test.<br><br>
    test.py<br>               # Creates the configuration files.<br><br>
    visualization.py<br>      # Contains functions that are used on the visualization of the Neural Network results and predictions.<br><br>
    experiments.ipynb<br>     # The python notebook that we run on colab.<br><br>
    *.sh<br>                  # Scripts to run fast the executables. Used during the development.<br><br>
    
</h4>

<h2>Tasks</h2>

<h3>Task A</h3>
<h4>"Build an image autocoding neural network that includes layers
compression and decompression (“bottleneck”). You will need to perform his training experiments
network with different hyperparameter values ​​[number of cohesive layers, size
cohesive filters, number of cohesive filters per layer, number of training seasons
(epochs), batch size, compression dimension (latent dimension, default = 10)] to
minimize the loss by avoiding overfitting. His data
entrance set must be properly divided into training set and training set
validation (validation set). Based on the experiments, you select the optimal structure for the neural network,
and the latent vector is used to represent the images in the new one
vector space."</h4>
<h4>
Execute: python reduce.py –d 'dataset' -q 'queryset' -od 'output_dataset_file' -oq
'output_query_file'</h4>
<h3>Task B</h3>
<h4>"Expand and use the deliverable of the first job to find the nearest one
neighbor of the search set images in the new vector space (exhaustive search)
as well as the true (LSH) and nearest (LSH) nearest neighbor to the original
vector space: all searches are done with metric Manhattan. Results
compared to the search time and the approach fraction in the original space, i.e. the mean
distance Manhattan proximal (NeuralNet or LSH) / actual nearest neighbor to
query vector in homepage."</h4>
<h4>
Execute the first task and use the outputs as the "new space files"
Execute: ./bin/search –d 'input file original space' -i 'input file new space' –q 'query file
original space' -s 'query file new space' –k 'int' -L 'int' -ο 'output file'</h4>
<h3>Task C</h3>
<h4>"Implement the Earth Mover’s Distance (EMD) metric reduced to solving a Linear problem
Linear Programming. Exhaustively find the 10 closest neighbors and
compare execution time and "correctness" versus exhaustive query search
B. The information provided by the image labels is used for "correctness". As a measure
The percentage of the nearest neighbors that have the same label as the image is set
question mark. Perform experiments on different sized clusters when calculating the distance
EMD and comment on the results in terms of time and "correctness"."</h4>
<h4>
Execute: ./emd/search_emd.py -d 'input file original space' –q 'query file original space' -l1<br>
'labels of input dataset' -l2 'labels of query dataset' -ο 'output file' -EMD</h4>
<h3>Task D</h3>
<h4>"Group S1 k-medians of the images of the input set in the new space and
let S2 be in the original space. Use of the deliverable of the 2nd task to categorize the images
of the input set and clustering S3 based on it. Extension and use of the deliverable of the 1st
to compare the three silhouettes in terms of silhouette and its evaluation
target function in the original space (k ~ 10) with Manhattan metric."</h4>
<h4>
Execute: ./bin/cluster –d 'input file original space' -i 'input file new space'
-n 'classes from NN as clusters file' –c 'configuration file' -o 'output file'</h4>

<h3>Assumptions</h3>
<h4>
1. Our PCs where unable to run experiments with the whole datasets, at least on C++, so we reduced it.<br>
2. For the same reason we executed the Python experiments on Google Colab to use more data.
</h4>
<h3>Experiments Details</h3>


