import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from stqdm import stqdm
import plotly.express as px



st.title("Color Clustering for Image Compression")
# menu = ["Image","Dataset","DocumentFiles","About"]
# choice = st.sidebar.selectbox("Menu",menu)
def cluster(image, cluster_type="nog"):
    print(cluster_type)
def load_image(image_file):
	img = Image.open(image_file)
	return img


from PIL import Image
from stqdm import stqdm
class ClusterImage:
    def __init__(self, image_obj):
        self.image_obj = image_obj
        self.shape = np.shape(np.array(self.image_obj))
        self.image_array = np.reshape(np.array(self.image_obj),(self.shape[0] * self.shape[1], self.shape[2]) )
        self.clustered_image_array = np.reshape(np.array(self.image_obj),(self.shape[0] * self.shape[1], self.shape[2]) )
        self.cluster_image_obj = None

    def get_clustered_image(self, clusters = 10, clustering_type = "KMeans"):
        match clustering_type:
            case "KMeans":
                # st.write('kmeans')
                kmeans = KMeans(n_clusters=clusters, random_state=0, max_iter = 10, n_init=1).fit(self.image_array)
                for i, pixel in enumerate(stqdm(self.image_array)):
                    cluster = kmeans.labels_[i]
                    self.clustered_image_array[i] = [int(samp) for samp in kmeans.cluster_centers_[cluster] ]
        
        self.cluster_image_obj = Image.fromarray(np.reshape(self.clustered_image_array, self.shape))

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
image_uploaded = False
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
        image_uploaded = True
        # To See details
        file_details = {"filename":image_file.name, "filetype":image_file.type,
                        "filesize":image_file.size}
        # st.write(file_details)

        # To View Uploaded Image
        st.image(load_image(image_file),width=250)
        cluster_image = ClusterImage((load_image(image_file)))
        st.write(f"Shape:{cluster_image.shape}")

selection = st.selectbox("Select clustering algorithm", ["KMeans", "GMM", "DBSCAN"], disabled= not image_uploaded)
num_clusters = st.slider("Number of Clusters", 1,50,1)
st.write(num_clusters)
if st.button("Begin Clustering", help="Start the color clustering algorithm", disabled= not image_uploaded):
    st.write("Clustering...")
    cluster_image.get_clustered_image(num_clusters,selection)
    st.write("Cluster Done")

    figure_unclustered = cluster_image.graph_color_space(clustered_colors=False)
    figure_clustered = cluster_image.graph_color_space(clustered_colors=True)
    col1, col2 = st.columns(2)
    col1.title("Original Image")
    col1.image(cluster_image.image_obj)
    col1.plotly_chart(figure_unclustered, use_container_width=True)
    col2.title("Clustered Image")
    col2.image(cluster_image.cluster_image_obj)
    col2.plotly_chart(figure_clustered, use_container_width=True)
    # plotly_chart(figure_clustered, use_container_width=False)
    # st.image(cluster_image.cluster_image_obj,width=250)
