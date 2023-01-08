import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras import preprocessing

st.title('Prostate Cancer Image Classification')

# file = st.file_uploader('Please upload an image file', type=['jpg', 'png', 'tif'])

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
  st.write(predictions)
  st.write(scores)
  result = f'{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence.' 
  return result


file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg",'tif'])
if file_uploaded is not None:    
  image = Image.open(file_uploaded)
  st.image(image, caption='Uploaded Image', use_column_width=True)
if file_uploaded is None:
  st.write("Invalid command, please upload an image")
else:
  with st.spinner('Model working....'):
    predictions = predict(image)
    st.success('Classified')
    st.write(predictions)

# if file is None:
#   st.text('Please upload an image file')
# else:
#   image = Image.open(file)
#   st.image(image, use_column_width=True)
#   predictions = import_and_predict(image, model)
#   score = tf.nn.softmax(predictions[0])
#   st.write(predictions)
#   st.write(score)
#   print(
#   'This image most likely belongs to {} with a {:.2f} percent confidence.'
#   .format(['Benign', 'G3', 'G4', 'G5'][np.argmax(score)], 100 * np.max(score))
# )


# pic = st.file_uploader('Upload an Image', type=['jpg', 'png', 'webp'])
# if pic:
#     st.image(pic, 'Your uploaded image')

#     with st.spinner('Detecting and analyzing faces....'):
#         pic = Image.open(pic)
#         pic = np.array(pic)
#         faces, age, gender = get_img_and_predict(pic)
#         if len(faces) == 0:
#             st.warning('No faces detected!')
#         else:
#             st.success('Success!! '+str(len(faces))+ ' faces detected and analyzed.')
#             cols = st.columns(len(faces))
#             for i in range(0, len(faces)):
#                 with cols[i]:
#                     st.image(faces[i])
#                     st.write('Age: ' + str(age[i]))
#                     st.write('Gender: ' + str(gender[i]))