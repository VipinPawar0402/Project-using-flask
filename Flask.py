import streamlit as st
import numpy as np
import os
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
#load model
model = load_model("model/v3_pred_cott_dis.h5")
print('@@ Model loaded')

def pred_cot_dieas(cott_plant):
    test_image = load_img(cott_plant, target_size = (150, 150)) # load image 
    print("@@ Got Image for prediction")
    
    test_image = img_to_array(test_image)/255 # convert image to np array and normalize
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

def main():
    st.title("Cotton Plant Disease Prediction App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = load_img(uploaded_file, target_size=(150, 150))
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label, page = pred_cot_dieas(uploaded_file)
        if label == 'Healthy Cotton Plant':
            st.success("Result: {}".format(label))
        else:
            st.error("Result: {}".format(label))
        html_file = open(page, 'r', encoding='utf-8')
        source_code = html_file.read()
        print(page)
        print(label)
        print(source_code)
        components.html(source_code)
    
if __name__ == "__main__":
    main()
