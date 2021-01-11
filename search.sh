./bin/search -d ./data/train-images-idx3-ubyte \
-i data/data_reduced_100 \
-q ./data/t10k-images-idx3-ubyte \
-s data/query_reduced_100 \
-k 4 \
-L 4 \
-o ./outputs/search_output.txt 
