import numpy as np


class Dataset(object):
    def __init__(self, voc_path, lib_path):
        """
        
        :param voc_path: 
        :param lib_path: 
        :return: 
        """
        self.num_lib = 0
        self.voc_size = 0
        self.x_ = []
        self.voc_dict = {}
        self.voc_path = voc_path
        self.lib_path = lib_path
        self.create_vocDict(voc_path)
        self.create_trainset(lib_path)
        # print voc_path

    def create_vocDict(self, voc_path):
        """
        
        :param voc_path: 
        :return: 
        """
        voc_file = open(voc_path)
        for line in voc_file.readlines():
            words = line.split('\t')
            self.voc_dict[int(words[0])] = words[1]
            self.voc_size += 1
        voc_file.close()

    def create_trainset(self, lib_path):
        """
        
        :param lib_path: 
        :return: 
        """
        lib_file = open(lib_path)
        for line in lib_file.readlines():
            self.num_lib += 1
            doc = np.zeros(self.voc_size, dtype='float')
            words = line.strip().split('\t')[1].split(" ")
            for word in words:
                word = word.split(":")
                doc[int(word[0])] = int(word[1])
            self.x_.append(doc)
        lib_file.close()
