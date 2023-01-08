import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing

st.title('Prostate Cancer Image Classification')
st.image(Image.open('avatar.webp'), caption='Gleason Score', use_column_width=True)

def predict(image):
  model = load_model('prostate_model_densenet.h5')
  test_image = image.resize((224,224))
  test_image = preprocessing.image.img_to_array(test_image)
  test_image = test_image / 255.0
  test_image = np.expand_dims(test_image, axis=0)
  class_names = ['Benign', 'G3', 'G4', 'G5']
  predictions = model.predict(test_image)
  scores = tf.nn.softmax(predictions[0])
  scores = scores.numpy()
  # st.write(predictions)
  # st.write(scores)
  result = f'Prostate Cancer: {class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence.' 
  return result


file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg",'tif'])
if file_uploaded is not None:    
  image = Image.open(file_uploaded)
  st.image(image, caption='Uploaded Image', use_column_width=True)
if file_uploaded is None:
  st.text("Please upload an image file")
else:
  with st.spinner('Model working....'):
    predictions = predict(image)
    st.success('Classified')
    st.write(predictions)