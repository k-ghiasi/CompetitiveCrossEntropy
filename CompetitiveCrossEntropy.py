"""

Author: Sayed Kamaledin Ghiasi-Shirazi
Year:   2019

"""

import numpy as np
import matplotlib.pyplot as plt

class CompetitiveCrossEntropy:
    def __init__ (self, trainingData, learning_rate, lr_decay_mult, max_epochs, weight_decay, noise_std=0):
        self.trainingData = trainingData
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.lr_decay_mult = lr_decay_mult
        self.noise_std = noise_std
        
        self.C  = trainingData.C
        self.X = trainingData.X
        self.y = trainingData.y
        self.L = trainingData.L
        self.label = trainingData.label
        self.K = trainingData.K
        self.subclassMeans = trainingData.subclassMeans

        self.b = np.zeros([self.L, 1])
        self.W = self.subclassMeans + 0
        for i in range(self.L):
            self.b[i] = -0.5 * np.linalg.norm(self.W[i, :]) ** 2

    def fit (self, reset=False):
        X = self.X
        N, dim = X.shape
        if reset:
            self.b= np.zeros([self.L,1])
            self.W = self.subclassMeans + 0
            for i in range (self.L):
                self.b[i] = -0.5 * np.linalg.norm(self.W[i,:]) ** 2
        bias = self.b
        W = self.W

        alpha = self.learning_rate
        iter = 0
        target_iter = N

        noise = np.random.randn(N, dim) * self.noise_std
        noise -= np.mean(noise, axis = 0)
        for i in range (self.max_epochs):
            shuffle = np.random.permutation(N)
            for j in range (N):
                n = shuffle[j]
                iter = iter + 1

                if (iter >= target_iter):
                    alpha = alpha * self.lr_decay_mult
                    iter = 0

                x = X[n,:] + noise[j,:]
                t = self.y[n]
                w = W @ x[:,None] + bias
                w = w[:,0]
                w = w - np.max(w)
                z = np.exp(w)
                denum = np.sum(z)
                if denum <= 10e-10:
                    denum = 10e-10
                z = z / denum

                tau = np.zeros(len(z))

                idx1 = t * self.K
                idx2 = (t+1) * self.K
                denum = np.sum (z[idx1:idx2])
                if (denum <= 10e-10):
                    denum = 10e-10
                tau[idx1:idx2] = z[idx1:idx2] / denum

                dE_dwk = z - tau
                bias = bias - alpha * dE_dwk[:,None]
                W = W - alpha * (np.outer(dE_dwk, x) + self.weight_decay * W)

        self.W = W
        self.b  = bias

    def classifyByMaxClassifier(self, Xtest):
        Xtest = Xtest
        A = self.W
        b = self.b
        d = A @ Xtest.T + b
        y_pred = self.label[d.argmax(axis=0)]
        return y_pred

    def transform (self, XTest):
        A = self.W
        return XTest @ A.T
                        
    def GenerateImagesOfWeights(self, width, height, color = 'color',
                        n_images=1, rows=None, cols=None, eps = 0):
        A = self.W
        nFeaturesPerImage = rows * cols  # (A.shape[0] + 1) // n_images
        if rows == None or cols == None:
            cols = int(np.sqrt(A.shape[0] - 1)) + 1
            rows = (A.shape[0] + cols - 1) // cols
        images = []
        for picture in range(n_images):
            img = np.ones([rows * (height + 1), cols * (width + 1), 3])
            for nn in range(nFeaturesPerImage):
                n = picture * nFeaturesPerImage + nn
                if (n >= A.shape[0]):
                    continue
                j = nn // rows
                i = nn % rows
                idx1 = i * (height + 1)
                idx2 = j * (width + 1)
                T = max(-np.min(A[n, :]), np.max(A[n, :])) + eps
                if color == 'color':
                    arr_pos = np.maximum(A[n,:] / T, 0)
                    arr_neg = np.maximum(-A[n,:] / T, 0)
                    mcimg_pos = np.reshape(arr_pos, [height, width])  
                    mcimg_neg = np.reshape(arr_neg, [height, width])  
                    mcimg_oth = 0
                elif color == 'gray':
                    arr = A[n, :] / (2 * T) + 0.5              
                    mcimg_pos = np.reshape(arr, [height, width])
                    mcimg_neg = mcimg_pos
                    mcimg_oth = mcimg_pos
                else:
                    assert (0)
                    
                img[idx1:idx1 + height, idx2:idx2 + width, 0] = mcimg_pos
                img[idx1:idx1 + height, idx2:idx2 + width, 1] = mcimg_neg
                img[idx1:idx1 + height, idx2:idx2 + width, 2] = mcimg_oth
            images.append(img)
        return images
