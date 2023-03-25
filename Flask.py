import streamlit as st
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("v3_pred_cott_dis.h5")
    return model

model = load_model()
   
def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size = (150, 150)) # load image 
    print("@@ Got Image for prediction")
   
    test_image = img_to_array(test_image)/150 # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
    result = model.predict(test_image).round(3) # predict diseased palnt or not
    print('@@ Raw result = ', result)
   
    pred = np.argmax(result) # get the index of max value
 
    if pred == 0:
        return "Healthy Cotton Plant", 'healthy_plant_leaf.html' # if index 0 burned leaf
    elif pred == 1:
        return 'Diseased Cotton Plant', 'disease_plant.html' # # if index 1
    elif pred == 2:
        return 'Healthy Cotton Plant', 'healthy_plant.html'  # if index 2  fresh leaf
    else:
        return "Healthy Cotton Plant", 'healthy_plant.html' # if index 3

     
# Create streamlit app
def app():
    st.set_page_config(page_title="Cotton Plant Disease Prediction App")

    st.title("Cotton Plant Disease Prediction App")

    # Display image uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded image temporarily to disk
        image_dir = "temp"
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        file_path = os.path.join(image_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Predict the class of the uploaded image and display the result
        st.write("Predicting...")
        pred, output_page = pred_cot_dieas(cott_plant=file_path)
        st.write(f"Prediction: {pred}")
        output_html = open(output_page).read()
        st.components.v1.html(output_html, height=600)

    
if __name__ == "__main__":
    app()
