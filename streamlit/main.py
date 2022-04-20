import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from stqdm import stqdm
import plotly.express as px
import random
from PIL import Image
from stqdm import stqdm
import pyautogui
# import hdbscan

st.title("Color Clustering for Image Compression")
# menu = ["Image","Dataset","DocumentFiles","About"]
# choice = st.sidebar.selectbox("Menu",menu)
def cluster(image, cluster_type="nog"):
    print(cluster_type)
def load_image(image_file):
	img = Image.open(image_file)
	return img



class ClusterImage:
    def __init__(self, image_obj):
        self.image_obj = image_obj
        self.shape = np.shape(np.array(self.image_obj))
        self.image_array = np.reshape(np.array(self.image_obj),(self.shape[0] * self.shape[1], self.shape[2]) )
        self.clustered_image_array = np.reshape(np.array(self.image_obj),(self.shape[0] * self.shape[1], self.shape[2]) )
        self.cluster_image_obj = None
        self.num_colors = None
        self.num_colors_clustered = None

    def get_clustered_image(self, clusters = 10, clustering_type = "KMeans(SciKit)"):
        match clustering_type:
            case "KMeans(SciKit)":
                # st.write('kmeans')
                kmeans = KMeans(n_clusters=clusters, random_state=0, max_iter = st.session_state['maxiterations_kmeans'], n_init=1).fit(self.image_array)
                for i, pixel in enumerate(stqdm(self.image_array,desc="Generating new image...")):
                    cluster = kmeans.labels_[i]
                    self.clustered_image_array[i] = [int(samp) for samp in kmeans.cluster_centers_[cluster] ]
            case "KMeans":
                # st.write("alex")
                # st.write(f"{clusters, self.shape[0], self.shape[1], self.image_obj.load()}")
                self.kmeans_(clusters, self.shape[1], self.shape[0], self.image_obj.load(), st.session_state['maxiterations_kmeans'])
            case "DBSCAN":
                self.dbscan_helper(st.session_state['DBSCAN_epsilon'], st.session_state['DBSCAN_min_samples'])

        
        self.cluster_image_obj = Image.fromarray(np.reshape(self.clustered_image_array, self.shape))
        self.num_colors =  np.unique(self.image_array, axis=0)
        self.num_colors_clustered = np.unique(self.clustered_image_array, axis=0)

    def kmeans_(self, kCount, width, height, px, maxIterations):
        # print(px)
        print("Running KMeans(alex)...")
        #creates an empty array for the centers
        centers = []

        #Randomly assigns "k" number of centers
        for k in stqdm(range(0, kCount)):
            randomX = random.randint(0, width-1)
            randomY = random.randint(0, height-1)
            center = px[randomX, randomY]
            centers.append(center)

        #main loop of the function
        """
        This section of the code will iterate 
        "maxIterations" times and assign RGB values
        to the pixels based on the current center.
        It will then update the centers
        """
        for i in stqdm(range(maxIterations), desc="Clustering Iterations"):
            # maxIterations = maxIterations - 1
            #creates the set of clusters of pixels based on the current centers
            clusters = {}
            #for each pixel in the image the loop will assign a cluster
            for x in stqdm(range(0, width)):
                for y in range(0, height):
                    pixelOfInterest = px[x, y]
                    minDistance = 9999999999
                    for i in range(0, len(centers)):
                        centerPoint = np.array((centers[i][0], centers[i][1], centers[i][2]))
                        pixelPoint = np.array((px[x,y][0], px[x,y][1], px[x,y][2]))
                        #utilize numpy euclidian distance formula
                        distanceToCenter = np.linalg.norm(centerPoint-pixelPoint)
                        if distanceToCenter < minDistance:
                            minDistance = distanceToCenter
                            associatedCenter = i
                    #assigns pixels to the cluster
                    if(associatedCenter in clusters.keys()):
                        clusters[associatedCenter].append(pixelOfInterest)
                    else:
                        clusters[associatedCenter] = [pixelOfInterest]

            #clears the centers array, populates new, more accurate centers
            centers = []
            keys = sorted(clusters.keys())
            #for each cluster, the new center is calculated based off the average color
            for k in keys:
                averageColor = np.mean(clusters[k], axis = 0)
                newCenter = (int(averageColor[0]), int(averageColor[1]), int(averageColor[2]))
                centers.append(newCenter)	

        #prints final centers				
        # print(centers)

        #for each pixel in the image, using the final centers, the RGB values are appended to a final output array
        #the RGB values are also the RBG value of the associated centers
        finalArray = []
        for x in stqdm(range(width), desc="Assigning final colors"):
            for y in range(height):
                minDistance = 9999999999
                associatedCenter = -1
                for i in range(0, len(centers)):
                        centerPoint = np.array((centers[i][0], centers[i][1], centers[i][2]))
                        pixelPoint = np.array((px[x,y][0], px[x,y][1], px[x,y][2]))
                        distanceToCenter = np.linalg.norm(centerPoint-pixelPoint)
                        if distanceToCenter < minDistance:
                            minDistance = distanceToCenter
                            associatedCenter = i
                finalArray.append(centers[associatedCenter])

        #formats the final array into a numpy array
        pixelArray = np.array(finalArray, ndmin= 2, dtype=np.uint8)
        pixelArray = np.reshape(pixelArray, (width, height, 3))
        pixelArray = np.rot90(pixelArray)
        pixelArray = np.rot90(pixelArray)
        pixelArray = np.rot90(pixelArray)
        pixelArray = np.fliplr(pixelArray)

        #creates and saves a new image using the numpy array
        self.clustered_image_array = np.reshape(pixelArray, (self.shape[0]* self.shape[1], 3))
        newImage = Image.fromarray(pixelArray)
        # newImage.save('test.png')
        #check to make sure the final array equals image width * image height
        # print(len(finalArray))
        self.cluster_image_obj = newImage
    
    def dbscan_helper(self, epsilon, min_samples):
        from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
        db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(self.image_array)
        labels = db.labels_
        cluster_values = {}
        for l in stqdm(labels, desc="Collecting clusters"):
            if l in cluster_values.keys():
                np.append(cluster_values[l], self.image_array[l])
                pass
            else:
                cluster_values[l] = [self.image_array[l]]

        cluster_values_avg = {}
        for l in stqdm(labels, desc="Averaging cluster members"):
            if l in cluster_values.keys():
                # print(cluster_values[l])
                cluster_values_avg[l] = np.mean(cluster_values[l], axis=0)

        for i, label in enumerate(stqdm(labels,desc="Generating new image...")):
            self.clustered_image_array[i] = [int(samp) for samp in cluster_values_avg[label] ]


        self.cluster_image_obj = Image.fromarray(self.clustered_image_array)



    def graph_color_space(self, clustered_colors=False):
        if clustered_colors:   
            image_info = self.clustered_image_array
            TITLE = "Reduced Image Colorspace"
        else:
            image_info = self.image_array
            TITLE = "Original Image Colorspace"
        colors = np.unique(image_info, axis=0)
        color_names = []
        for col in (colors):
            color_names.append('Unknown Color')
        data = list(map(list, zip(colors[:,0], colors[:,1],colors[:,2],color_names)))
        new_df = pd.DataFrame(data,columns=['Red','Green','Blue', "Name"])
        colfig = px.scatter_3d(new_df, x='Red', y='Green', z='Blue',
                    color='Blue', title= TITLE)
        return colfig
if 'image_uploaded' not in st.session_state:
    st.session_state['image_uploaded'] = False
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
    st.session_state['image_uploaded']  = True
    # # To See details
    # file_details = {"filename":image_file.name, "filetype":image_file.type,
    #                 "filesize":image_file.size}
    # # st.write(file_details)

    # To View Uploaded Image
    st.image(load_image(image_file),use_column_width='always')
    cluster_image = ClusterImage((load_image(image_file)))
        # st.write(f"Shape:{cluster_image.shape}")

st.session_state['selection'] = st.selectbox("Select clustering algorithm", ["KMeans(SciKit)", "KMeans", "DBSCAN"], disabled= not st.session_state['image_uploaded'] )
if st.session_state['image_uploaded']:  
    
    match st.session_state['selection']:
        case "KMeans" | "KMeans(SciKit)":
            st.session_state['num_clusters'] = st.slider("Number of Clusters", 1,50,1)
            st.session_state['maxiterations_kmeans'] = st.slider("Number of iterations", 1, 15, 3)
        case "DBSCAN":
            # st.session_state['DBSCAN_active'] = True
            st.session_state['DBSCAN_min_samples'] = st.slider("Minimum Samples", 5,20,1,help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.")
            st.session_state['DBSCAN_epsilon'] = st.slider("Epsilon Value", 0.5,2.0,0.5,0.1 ,help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
# st.write(num_clusters)
st.session_state['BeginCluster'] = st.button("Begin Clustering", help="Start the color clustering algorithm", disabled= not st.session_state['image_uploaded'])

if st.session_state['BeginCluster']:
    # st.write("Clustering...")
    cluster_image.get_clustered_image(st.session_state['num_clusters'],st.session_state['selection'])
    # st.write("Clustering Complete")

    figure_unclustered = cluster_image.graph_color_space(clustered_colors=False)
    figure_clustered = cluster_image.graph_color_space(clustered_colors=True)
    col1, col2 = st.columns([2,2])
    col1.title("Original Image")
    col1.image(cluster_image.image_obj)
    col1.plotly_chart(figure_unclustered, use_container_width=True)
    col2.title("Clustered Image")
    col2.image(cluster_image.cluster_image_obj)
    col2.plotly_chart(figure_clustered, use_container_width=True)
    # plotly_chart(figure_clustered, use_container_width=False)
    # st.image(cluster_image.cluster_image_obj,width=250)

if st.button("Reset"):
    pyautogui.hotkey("ctrl","F5")

