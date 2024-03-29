import tensorflow as tf
import numpy as np
import math
from numba import jit
import os

@jit(nopython=True)
def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

@jit(nopython=True)
def softmax(x: float) -> float:
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)

@jit(nopython=True)
def softmax_derivative(x: float) -> float:
    value = x * (1 - x)
    return value

@jit(nopython=True)
def cross_entropy_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ret = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] != 0 and x[i, j] != 1:
                ret[i, j] = (-x[i,j] + y[i,j]) / (x[i,j] * (x[i,j] - 1))
    
    return ret

@jit(nopython=True)
def convert(x: int) -> np.ndarray:
    arr = np.zeros((10, 1))
    arr[x][0] += 1
    return arr

@jit(nopython=True)
def _forward_propagation(x: np.ndarray, h1: np.ndarray, o: np.ndarray, weightIH1: np.ndarray, 
                         weightH2O: np.ndarray, biasH1: np.ndarray, biasO: np.ndarray) -> None:
    h1 += np.dot(weightIH1, x) + biasH1
    h1 *= (h1 > 0)
    
    o += softmax(np.dot(weightH2O, h1) + biasO)
    
@jit(nopython=True)
def _backward_propagation(inp: np.ndarray, h1: np.ndarray, o: np.ndarray, exp: np.ndarray, weightIH1: np.ndarray, 
                          weightH2O: np.ndarray, biasH1: np.ndarray, biasO: np.ndarray, lr: float) -> None:
    dBO = cross_entropy_derivative(o, exp) * softmax_derivative(o)
    dWH2O = np.dot(dBO, h1.transpose())
    
    dB1 = np.dot(weightH2O.transpose(), dBO) * (h1 > 0)
    dWIH1 = np.dot(dB1, inp.transpose())
    
    weightIH1 -= dWIH1 * lr
    biasH1 -= dB1 * lr
    
    weightH2O -= dWH2O * lr
    biasO -= dBO * lr   

class NeuralNetwork:
    def __init__(self, toLoad: bool, ep: int = 100, batch: int = 10000, lr: float = 0.01, num_h1: int = 128, num_o: int = 10) -> None:
        np.random.seed(7)
        script_path = os.path.abspath(__file__)
        self.directory = os.path.dirname(script_path)
        mnist = tf.keras.datasets.mnist
        (train_images, self.train_labels), (test_images, self.test_labels) = mnist.load_data()    
        
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
        
        self.train_labels = self.train_labels.reshape((-1, 1))
        self.test_labels = self.test_labels.reshape((-1, 1))
        
        # Reshape each image as a vector
        self.vector_images = np.reshape(train_images, (train_images.shape[0], 784, 1)).astype(np.float64)
        self.test_images = np.reshape(test_images,(test_images.shape[0], 784, 1)).astype(np.float64)
        
        if toLoad:
            self.weightIH1 = np.load(self.directory + "/weightIH1.npy")
            self.weightH2O = np.load(self.directory + "/weightH2O.npy")
            
            self.biasH1 = np.load(self.directory + "/biasH1.npy")
            self.biasO = np.load(self.directory + "/biasO.npy")
            
        else:
            self.weightIH1 = np.random.random((num_h1, 784)).astype(np.float64) - 0.5
            self.weightH2O = np.random.random((num_o, num_h1)).astype(np.float64) - 0.5
            
            self.biasH1 = np.random.random((num_h1,1)).astype(np.float64) / 10
            self.biasO = np.random.random((num_o,1)).astype(np.float64) / 10
        
        self.h1 = np.zeros((num_h1,1)).astype(np.float64)
        self.o = np.zeros((num_o,1)).astype(np.float64)
        
        self.ep = ep
        self.batch = batch
        self.lr = lr
    
    
    def get_ans(self) -> int:
        max_val = -math.inf
        index = -1
        for i in range(self.o.shape[0]):
            if self.o[i][0] > max_val:
                max_val = self.o[i][0]
                index = i
        
        return index
    
    def train(self) -> None:
        index = 0
        indices = np.arange(self.vector_images.shape[0])
        np.random.shuffle(indices)
        imgs = self.vector_images[indices]
        labels = self.train_labels[indices]
        for epoch in range(self.ep):
            if(index >= self.vector_images.shape[0]):
                index = 0
                indices = np.arange(self.vector_images.shape[0])
                np.random.shuffle(indices)
                imgs = self.vector_images[indices]
                labels = self.train_labels[indices]
            
            for i in range(self.batch):
                inp : np.ndarray = np.copy(imgs[index])
                inp[inp == 0] = 0.0001
                _forward_propagation(inp, self.h1, self.o, self.weightIH1, self.weightH2O, self.biasH1, self.biasO)
                
                _backward_propagation(inp, self.h1, self.o, convert(labels[index][0]), self.weightIH1, self.weightH2O, self.biasH1, self.biasO, self.lr)
                
                self.o = np.zeros_like(self.o)
                self.h1 = np.zeros_like(self.h1)
                
                index += 1
                if(index == self.vector_images.shape[0]):
                    break
            
            if epoch == self.ep // 2:
                self.lr /= 4
            
            print("epoch {0} percentage test {1}".format(epoch + 1, self.evaluate()))
        
        np.save(self.directory + "/weightIH1.npy", self.weightIH1)
        np.save(self.directory + "/weightH2O.npy", self.weightH2O)
        
        np.save(self.directory + "/biasH1.npy", self.biasH1)
        np.save(self.directory + "/biasO.npy", self.biasO)
    
    def overfit(self) -> float:
        correct= 0
        for i in range(self.vector_images.shape[0]):
            _forward_propagation(self.vector_images[i], self.h1, self.o, self.weightIH1, self.weightH2O, self.biasH1, self.biasO)
            ans = self.get_ans()
            
            if ans == self.train_labels[i][0]:
                correct += 1
        
            self.o = np.zeros_like(self.o)
            self.h1 = np.zeros_like(self.h1)
        
        return (correct / self.vector_images.shape[0]) * 100
    
    def evaluate(self) -> float:
        correct= 0
        for i in range(self.test_images.shape[0]):
            _forward_propagation(self.test_images[i], self.h1, self.o, self.weightIH1, self.weightH2O, self.biasH1, self.biasO)
            ans = self.get_ans()
            
            if ans == self.test_labels[i][0]:
                correct += 1
        
            self.o = np.zeros_like(self.o)
            self.h1 = np.zeros_like(self.h1)
        
        return (correct / self.test_images.shape[0]) * 100
    
    def activate(self, inp: np.ndarray):
        inp = np.reshape(inp, (784, 1)).astype(np.float64)
        mod_inp = np.copy(inp)
        mod_inp[mod_inp == 0] = 0.0001
        _forward_propagation(mod_inp, self.h1, self.o, self.weightIH1, self.weightH2O, self.biasH1, self.biasO)
        ans = self.get_ans()
        
        self.o = np.zeros_like(self.o)
        self.h1 = np.zeros_like(self.h1)
        
        print("ARI guess it's a {0}".format(ans))
            
 
if __name__ == "__main__":
    nn = NeuralNetwork(False, 40, 10000, 0.01, 128, 10)
    nn.train()