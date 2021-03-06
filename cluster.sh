./bin/cluster -d data/train-images-idx3-ubyte \
-i data/data_reduced_10  \
-n data/clusters10000.txt \
-c configurations/cluster.conf \
-o outputs/cluster_output.txt


# valgrind -v --trace-children=yes --show-leak-kinds=all --leak-check=full  --fair-sched=yes --track-origins=yes \
# ./bin/cluster -d data/train-images-idx3-ubyte \
# -i data/data_reduced_10  \
# -n data/clusters.txt \
# -c configurations/cluster.conf \
# -o outputs/cluster_output.txt

# gdb --args ./bin/cluster -d data/train-images-idx3-ubyte \
# -i data/data_reduced_10  \
# -n data/clusters.txt \
# -c configurations/cluster.conf \
# -o outputs/cluster_output.txt


