import streamlit as st
import numpy as np
import os

@st.cache(allow_output_mutation=True)
def load():
    return load_model("v3_pred_cott_dis.h5")

model = load()
print('@@ Model loaded')

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
  st.set_page_config(page_title="Cotton Plant Disease Prediction", page_icon=":corn:", layout="wide")
  st.title("Cotton Plant Disease Prediction")
  st.markdown("Upload a picture of a cotton plant leaf and get a prediction on whether it is healthy or diseased.")
  st.markdown("---")
  uploaded_file = st.file_uploader("Choose a cotton plant leaf image...", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
    image = np.array(load_img(uploaded_file, target_size=(150, 150)))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, output_page = pred_cot_dieas(uploaded_file)
    st.success(f"The cotton plant is **{label}**")
    if label == "Healthy Cotton Plant":
      st.warning("This is just a precaution. Do consult an agriculturalist if the condition persists.")
    st.markdown("---")
    st.markdown(f"See the [source code]({os.path.basename(__file__)}) of this app.")

if __name__ == "__main__":
    app()  
