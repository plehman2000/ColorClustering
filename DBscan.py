import time
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from matplotlib.pyplot import imshow
#plot
def plot(C_cluster, d_data):
    c_ = ['r', 'g', 'b']
    l_ = ['group_{}'.format(i+1) for i in range(3)]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i,g in enumerate(C_cluster):
        ax.scatter(*zip(*d_data[list(g)]), c=c_[i],label=l_[i])
        plt.legend(loc='lower right')
    plt.show()

#dataset will be replace with get image
def load_data():
    d = load_iris()
    # print(d.data.shape)
    # print(d.target)
    k = len(np.unique(d.target))
    assert k == 3

    # Sample Data
    idx = list(range(5)) + list(range(50, 55)) + list(range(100, 105))
    d_data = d.data[idx, 2:]  # only select 15 samples and two features
    d_target = d.target[idx]
    # print(d_targe) # choose 5 from each class

    return d_data, d_target, k

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


def get_C_ref(C_cluster, d_target, k):
    default_C = np.array_split(range(15), 3)
    C_ref = []
    for i in range(k):
        c = np.zeros(shape=k)
        for j in C_cluster[i]:
            c[d_target[j]] += 1
        L = np.argmax(c)
        C_ref.append(set(default_C[L]))
        return C_ref

#Distance

def Euclidean(x1, x2, p=2): #works
        return np.sum(np.abs(x1-x2)**p)**float(1/p)

def FindDist(p1, p2):
    dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**float(1/2)
    return dist

#DBSCAN
def my_Find_Neighbor(imgData, pair, epsilon):
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

def DBSCAN(imgData, epsilon, MinPts=3, max_iter = 100):
    H = len(imgData)
    W = len(imgData[0])
    omega = set()                   #2d set holding [H][W] of pixel (Primaries)
    Nbr_all = [[0 for x in range(W)] for y in range(H)]                #Holds Pixel coords of all
    for i in range(0, H):
        for j in range(0, W):
            pair = (i, j)
            Nbr = my_Find_Neighbor(imgData, pair, epsilon)    #array of Neighbor coords
            Nbr_all[i][j] = Nbr                                     #array of Nbr arrays
            if len(Nbr) >= MinPts:
                omega.add(pair)
        print(i)
    #print(H*W)
    #print(len(omega))

    cluster = 0                 #Number of Clusters
    unvisited = [[True for x in range(W)] for y in range(H)]

    ClusterArr = [[0 for x in range(W*H)] for y in range(max_iter)]
    unvisited_old = [[True for x in range(W)] for y in range(H)]
    while(len(omega) != 0):
        #print("A")
        for i in range(0, len(unvisited)):
            for j in range(0, len(unvisited[0])):
                unvisited_old[i][j] = unvisited[i][j]
        #unvisited_old = unvisited.copy()
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
                    #print("Range: ", len(Nbr_all[curCenter[0]][curCenter[1]]))
                    #print("Test: ", s)
                    if unvisited[s[0]][s[1]]:
                        delta.append(s)
                        unvisited[s[0]][s[1]] = False
        Pairs = []
        for i in range(0, len(unvisited)):
            for j in range(0, len(unvisited[i])):
                #print("accessed")
                #print("New: ", unvisited[i][j])
                #print("Old: ", unvisited_old[i][j])
                if unvisited[i][j] != unvisited_old[i][j]:
                    tPair = (i, j)
                    Pairs.append(tPair)
                    #print("P: ", Pairs)

        cluster += 1
        cc += 1
        c_k = []
        if (Pairs != []):
            c_k.append(Pairs)
        for i in range(0, len(c_k)):
            for j in range(0, len(c_k[i])):
                #print("Trying to remove: ", c_k[i][j])
                if c_k[i][j] in omega:
                    omega.remove(c_k[i][j])
        ClusterArr.append(c_k)
        print(cluster)
    return ClusterArr

def ColorReturn(Clusters, ImgData):
    R = 0
    G = 0
    B = 0
    #print(Clusters[0][0], " ", Clusters[0][1])
    for i in range(0, len(Clusters)):
        R += ImgData[Clusters[i][0]][Clusters[i][1]][0]
        G += ImgData[Clusters[i][0]][Clusters[i][1]][1]
        B += ImgData[Clusters[i][0]][Clusters[i][1]][2]
    R = R/len(Clusters)
    G = G/len(Clusters)
    B = B/len(Clusters)
    RGB = (R, G, B)
    return RGB

def ImageReturn(Clusters, ImgData, RGB):
    h = len(ImgData)
    w = len(ImgData[0])
    output = np.arange(0, h*w, 1, np.uint8)
    for i in h:
        for j in w:
            print ("hi")


#main
def __main__(algo='DBSCAN'):
    #test Euclidian:
    file = Load_Data("rain.jpg")
    image_array = file.load()
    newCluster = np.array(DBSCAN(image_array,epsilon=5,MinPts=30))
    ClusterColorInfo = []
    inc = 0;
    for i in range(0, len(newCluster)):
        for j in range(0, len(newCluster[i])):
            if newCluster[i][0] != 0 or len(newCluster[i]) == 0:
            #print(newCluster[i])
                #revisedCluster[inc][j] = newCluster[i][j]
                RGB = ColorReturn(newCluster[i][j], image_array)
                ClusterColorInfo.append(RGB)
                inc += 1
                print(RGB)


__main__()