import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def segment_image(image, k):
    # Convertir l'image en un tableau numpy
    data = np.array(image)

    # Redimensionner l'image en un tableau 2D pour le clustering
    data_2D = data.reshape(-1, 3)

    # Appliquer K-Means sur l'image
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(data_2D)

    # Définir une palette de couleurs pour les clusters
    colors = np.random.randint(0, 255, size=(k, 3))

    # Remplacer chaque pixel par la couleur du centroïde du cluster correspondant
    segmented_image_array = np.zeros_like(data_2D)
    for i in range(k):
        color = colors[i]
        segmented_image_array[kmeans.labels_ == i] = color

    # Redimensionner l'image segmentée à sa taille d'origine
    segmented_image = segmented_image_array.reshape(data.shape)
    segmented_image = segmented_image.astype(np.uint8)

    return segmented_image

st.title("Segmentation d'image avec K-Means")

uploaded_image = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Image originale", use_column_width=True)

    k = st.slider("Nombre de clusters (K)", min_value=2, max_value=20, value=5)

    if st.button("Segmenter"):
        segmented_image = segment_image(image, k)
        st.image(segmented_image, caption="Image segmentée", use_column_width=True)
