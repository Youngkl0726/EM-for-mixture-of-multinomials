import numpy as np
import time

def log_sum(mat, axis):
    """
    
    :param mat: 
    :param axis: 
    :return: 
    """
    if axis==1:
        max_ = np.max(mat, axis=1).reshape(-1,1)
        return max_+np.log(np.sum(np.exp(mat-max_),axis=1).reshape(-1,1))
    else:
        max_ = np.max(mat, axis=0).reshape(1,-1)
        return max_+np.log(np.sum(np.exp(mat-max_),axis=0).reshape(1,-1))

class model(object):
    sim = 0.0
    def __init__(self,k,n,w):
        """
        
        :param k: k clusters
        :param n: n samples
        :param w: vocabulary size
        :return: 
        """
        self.k = k
        self.n = n
        self.w = w
        self.PI = np.random.rand(self.k)
        self.PI = self.PI/np.sum(self.PI)
        self.MU = np.random.rand(self.k,self.w)
        self.MU = self.MU/np.sum(self.MU,axis=1).reshape(-1, 1)


    def Comp_resp(self,x_,mu_,pi_):
        """
        
        :param x_: 
        :param mu_: 
        :param pi_: 
        :return: 
        """
        mat_log = np.dot(x_,np.log(mu_.T))+np.log(pi_)
        dominator = log_sum(mat_log, axis=1)
        a = mat_log-dominator
        return np.exp(a)

    def Comp_pi_and_mu(self,x_,Gamma_,n_):
        """
        
        :param x_: training set data
        :param Gamma_: Gamma(n,k) responsiblity of sample n to cluster k
        :param n_: sample num
        :return: refresh pi,mu
        """
        tmp = np.dot(Gamma_.T,x_)+1e-30
        mu = tmp/np.sum(tmp,axis=1).reshape(-1,1)
        pi = np.sum(Gamma_,axis=0)/n_
        return pi,mu

    def train(self,x_,iter_num):
        """
        
        :param x_: training set data
        :param iter_num: 
        :return: 
        """
        for i in range(iter_num):
            start_time=time.time()
            # E step
            self.Gamma = self.Comp_resp(x_=x_,mu_=self.MU,pi_=self.PI)
            #M step
            self.PI,self.MU = self.Comp_pi_and_mu(x_=x_,Gamma_=self.Gamma,n_=self.n)

            iter_duration=time.time()-start_time
            # print "Iteration %d, %f seconds." %(i,iter_duration)
        print "Training over."

    def output(self, num, dataset):
        """
        
        :param num: the num of words which rank up to print
        :param dataset: 
        :return: 
        """
        word = [[] for i in range(self.k)]
        f = file('./nips/output.txt', 'w+')
        print "The K is: %d" %(self.k)
        for i in range(self.k):
            print "The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i+1, self.PI[i], num)
            print>>f, "The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i+1, self.PI[i], num)
            list = []
            for j in range(self.w):
                list.append((self.MU[i,j],j))
            list.sort(key=lambda x:x[0], reverse=True)
            for j in range(num):
                word[i].append(dataset.voc_dict[list[j][1]])
                print dataset.voc_dict[list[j][1]],
                print>>f, dataset.voc_dict[list[j][1]]
            print ""


        # for i in range(5):
        #     for j in range(5):
        #         print word[i][j],
        #     print ""

        model.sim=0.0
        num = 0.0
        for i in range(self.k-1):
            for j in range(i+1,self.k):
                cnt = 0.0
                for k in range(5):
                    for l in range(5):
                        if word[i][k] is word[j][l]:
                            cnt = cnt+1.0
                # print cnt
                num = num+1.0
                model.sim = model.sim+cnt / 5.0
        model.sim = model.sim/num
        print "Average similarity is %f " %(model.sim)
        print >>f,"Average similarity is %f " %(model.sim)