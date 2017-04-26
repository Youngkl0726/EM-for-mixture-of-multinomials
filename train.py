import numpy as np
import Input
import model


# cluster_num = 20
mn = 1.0
num = [5,10,20,30]
index = 0

for i in range(4):
    if i>0:
        print ""
    cluster_num = num[i]
    voc_file = "./nips/nips.vocab"
    lib_file = "./nips/nips.libsvm"

    # file = open(lib_file)
    print "Geting dataset......"
    Dataset = Input.Dataset(voc_path=voc_file, lib_path=lib_file)
    print "Dataset done......"

    print "Building model......"
    MoM = model.model(cluster_num, Dataset.num_lib, Dataset.voc_size)
    print "Model done......"

    MoM.train(x_=np.array(Dataset.x_, dtype='float'), iter_num=20)

    MoM.output(5, dataset=Dataset)

    if MoM.sim < mn:
        mn = MoM.sim
        index = i
print ""
print "The best K is: %d " %(num[index])



# voc_file = "./nips/nips.vocab"
# lib_file = "./nips/nips.libsvm"
#
# # file = open(lib_file)
# print "Geting dataset......"
# Dataset = Input.Dataset(voc_path=voc_file, lib_path=lib_file)
# print "Dataset done......"
#
# print "Building model......"
# MoM = model.model(cluster_num, Dataset.num_lib, Dataset.voc_size)
# print "Model done......"
#
# MoM.train(x_=np.array(Dataset.x_, dtype='float'), iter_num=20)
# MoM.output(5, dataset=Dataset)
