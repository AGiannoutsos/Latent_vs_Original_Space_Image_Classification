# Latent vs Original Space Image Classification
  
<p style="text-align: center;">
    <img src="./doc/di_uoa.png" alt="UOA">
    <h1>University of Athens</h1>
    <h2>Department of Informatics and Telecomunications</h2>
</p>

<h3>Dionysis Taxiarchis Balaskas - 1115201700094</h3>
<h3>Andreas Giannoutsos - 1115201700021</h3>
<br>


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
<h4>"Κατασκευάστε νευρωνικό δίκτυο αυτοκωδικοποίησης εικόνων το οποίο θα περιλαμβάνει στρώματα
συμπίεσης και αποσυμπίεσης (“bottleneck”). Θα πρέπει να πραγματοποιήσετε πειράματα εκπαίδευσης του
δικτύου με διαφορετικές τιμές υπερπαραμέτρων [αριθμού συνελικτικών στρωμάτων, μεγέθους
συνελικτικών φίλτρων, αριθμού συνελικτικών φίλτρων ανά στρώμα, αριθμού εποχών εκπαίδευσης
(epochs), μεγέθους δέσμης (batch size), διάστασης συμπίεσης (latent dimension, default=10)] ώστε να
ελαχιστοποιήσετε το σφάλμα (loss) αποφεύγοντας την υπερπροσαρμογή (overfitting). Τα δεδομένα του
συνόλου εισόδου πρέπει να χωριστούν κατάλληλα σε σύνολο εκπαίδευσης (training set) και σε σύνολο
επικύρωσης (validation set). Βάσει των πειραμάτων, επιλέγετε τη βέλτιστη δομή για το νευρωνικό δίκτυο,
και το διάνυσμα συμπίεσης (latent vector) χρησιμοποιείται για την αναπαράσταση των εικόνων στον νέο
διανυσματικό χώρο"</h4>
<h4>
Execute: python reduce.py –d <dataset> -q <queryset> -od <output_dataset_file> -oq
<output_query_file></h4>
<h3>Task B</h3>
<h4>"Επεκτείνετε και χρησιμοποιείστε το παραδοτέο της πρώτης εργασίας για την εύρεση του πλησιέστερου
γείτονα των εικόνων του συνόλου αναζήτησης στον νέο διανυσματικό χώρο (εξαντλητική αναζήτηση)
καθώς και του πραγματικού (true) και του προσεγγιστικού (LSH) πλησιέστερου γείτονα στον αρχικό
διανυσματικό χώρο: όλες οι αναζητήσεις γίνονται με τη μετρική Manhattan. Τα αποτελέσματα
συγκρίνονται ως προς τον χρόνο αναζήτησης και το κλάσμα προσέγγισης στον αρχικό χώρο, δηλ. τη μέση
απόσταση Manhattan προσεγγιστικού (NeuralNet ή LSH) / πραγματικού πλησιέστερου γείτονα από το
διάνυσμα επερώτησης στον αρχικό χώρο."</h4>
<h4>
Execute the first task and use the outputs as the <new space files>
Execute: ./bin/search –d <input file original space> -i <input file new space> –q <query file
original space> -s <query file new space> –k <int> -L <int> -ο <output file></h4>
<h3>Task C</h3>
<h4>"Υλοποιήστε τη μετρική Earth Mover’s Distance (EMD) που ανάγεται σε επίλυση προβλήματος Γραμμικού
Προγραμματισμού (Linear Programming). Βρείτε εξαντλητικά τους 10 πλησιέστερους γείτονες και
συγκρίνετε τον χρόνο εκτέλεσης και την «ορθότητα» έναντι της εξαντλητικής αναζήτησης του ερωτήματος
Β. Για την «ορθότητα» χρησιμοποιείται η πληροφορία που δίνουν τα labels των εικόνων. Ως μέτρο
ορθότητας ορίζεται το ποσοστό των πλησιέστερων γειτόνων που έχουν το ίδιο label με την εικόνα
επερώτησης. Εκτελέστε πειράματα για διαφορετικό μέγεθος clusters κατά τον υπολογισμό της απόστασης
EMD και σχολιάστε τα αποτελέσματα ως προς τον χρόνο και την «ορθότητα»."</h4>
<h4>
Execute: ./emd/search_emd.py -d <input file original space> –q <query file original space> -l1
<labels of input dataset> -l2 <labels of query dataset> -ο <output file> -EMD</h4>
<h3>Task D</h3>
<h4>" Πραγματοποιήστε συσταδοποίηση Σ1 k-medians των εικόνων του συνόλου εισόδου στον νέο χώρο και
έστω Σ2 στον αρχικό χώρο. Χρήση του παραδοτέου της 2ης εργασίας για κατηγοριοποίηση των εικόνων
του συνόλου εισόδου και συσταδοποίηση Σ3 βάσει αυτής. Επέκταση και χρήση του παραδοτέου της 1ης
εργασίας για σύγκριση των τριών συσταδοποιήσεων ως προς το silhouette και την αποτίμηση της
συνάρτησης-στόχου στον αρχικό χώρο (k ~ 10) με μετρική Manhattan."</h4>
<h4>
Execute: ./bin/cluster –d <input file original space> -i <input file new space>
-n <classes from NN as clusters file> –c <configuration file> -o <output file></h4>

<h3>Assumptions</h3>
<h4>
1. Our PCs where unable to run experiments with the whole datasets, at least on C++, so we reduced it.
2. For the same reason we executed the Python experiments on Google Colab to use more data.
</h4>
<h3>Experiments Details</h3>


