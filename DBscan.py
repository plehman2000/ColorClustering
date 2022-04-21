import time
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

class Load_Data:
    def __init__(self, filename):
        self.image_obj = Image.open(filename).convert('RGB')
        self.shape = np.shape(np.array(self.image_obj))
        self.image_array = (np.array(self.image_obj))
        self.clustered_image_array = None
        self.cluster_image_obj = None
        #print(self.image_array[20][40][0], self.image_array[20][40][1], self.image_array[20][40][2])
        #print(len(self.image_array))
        #imshow(self.image_obj)
        #plt.show()
        #image_array = Height
        #image_array[0] = Width
        #image_array[0][0] = RGB
    def load(self):
        return self.image_array

class Dist:
    def Euclidean(x1, x2, p=2):
        return np.sum(np.abs(x1-x2)**p)**float(1/p)

def FindDist(p1, p2):
    dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**float(1/2)
    return dist

#DBSCAN
def my_Find_Neighbor(imgData, pair, dist, epsilon):
    Nbr = set()
    centerPnt = imgData[pair[0]][pair[1]]
    #print(epsilon)
    yl = pair[0] - epsilon
    xl = pair[1] - epsilon
    yh = pair[0] + epsilon
    xh = pair[1] + epsilon
    while(yl <= 0):
        yl += 1
    while(xl <= 0):
        xl += 1
    while(yh > len(imgData)):
        yh -= 1
    while (xh > len(imgData[0])):
        xh -= 1
    for i in range(yl, yh):
        for j in range(xl, xh):
            tempPair = (i,j)
            if tempPair != pair:
                RGBtemp = imgData[i][j]
                if(FindDist(RGBtemp, centerPnt)) < epsilon:
                    Nbr.add(tempPair)
    return Nbr



def myDBSCAN(imgData, epsilon=5, MinPts=3, max_iter = 100, tol=1e-20, diff_type='max', dist=Dist.Euclidean, name = "DBSCAN"):
    N = len(imgData)*len(imgData[0])
    H = len(imgData)
    W = len(imgData[0])
    omega = set()                   #2d set holding [H][W] of pixel (Primaries)
    Nbr_all = [[0 for x in range(W)] for y in range(H)]                #Holds Pixel coords of all
    for i in range(0, H):
        for j in range(0, W):
            pair = (i, j)
            Nbr = my_Find_Neighbor(imgData, pair, dist, epsilon)    #array of Neighbor coords
            Nbr_all[i][j] = Nbr                                     #array of Nbr arrays
            if len(Nbr) >= MinPts:
                omega.add(pair)
        print(i)
    print(H*W)
    print(len(omega))

    cluster = 0                 #Number of Clusters
    unvisited = [[True for x in range(W)] for y in range(H)]

    ClusterArr = [[0 for x in range(W*H)] for y in range(max_iter)]

    while(len(omega) != 0):
        print("A")

        unvisited_old = unvisited.copy()
        randIdxArr = []
        for i in range(10):
            randIdx = (np.random.randint(H), np.random.randint(W))
            randIdxArr.append(randIdx)
            unvisited[randIdx[0]][randIdx[1]] = False
            cc = 0
        while(len(randIdxArr) != 0):
            curCenter = randIdxArr[cc]
            randIdxArr.remove(curCenter)
            if len(Nbr_all[curCenter[0]][curCenter[1]]) >= MinPts:
                delta = []
                for s in Nbr_all[curCenter[0]][curCenter[1]]:
                    print("Range: ", len(Nbr_all[curCenter[0]][curCenter[1]]))
                    print("Test: ", s)
                    if unvisited[s[0]][s[1]]:
                        delta.append(s)
                        unvisited[s[0]][s[1]] = False
        Pairs = []
        for i in range(0, len(unvisited)):
            for j in range(0, len(unvisited[i])):
                #print("accessed")
                print("New: ", unvisited[i][j])
                print("Old: ", unvisited_old[i][j])
                if unvisited[i][j] != unvisited_old[i][j]:
                    tPair = (i, j)
                    Pairs.append(tPair)
                    print("P: ", Pairs)

        cluster += 1
        cc += 1
        c_k = []
        c_k.append(Pairs)
        for i in range(0, len(c_k)):
            print(c_k[i])
            tPair = (c_k[i][0],c_k[i][1])
            omega.remove(tPair)
        ClusterArr.append(c_k)
    return ClusterArr
    #ToAdd: Color Average, Image Compiler, Test Clusters





#main
def __main__(algo='DBSCAN'):

    #test Euclidian:
    file = Load_Data("rain.jpg")
    image_array = file.load()
    one = image_array[20][20]
    #print(one)
    two = image_array[40][20]
    myDBSCAN(image_array,epsilon=5,MinPts=20, dist=Dist.Euclidean)

    print("Distance: ", Dist.Euclidean(one,two,2))
    clusters = []
    arrOfClusters = []
    Hr = random.randrange(0,len(image_array))       #H
    Wr = random.randrange(0,len(image_array[0]))    #w
    print(image_array[Hr][Wr])
    print(two)
    print("Distance: ", Dist.Euclidean(image_array[Hr][Wr], two, 2))

__main__()