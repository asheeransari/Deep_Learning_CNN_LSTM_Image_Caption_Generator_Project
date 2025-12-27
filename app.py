from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model('model\new_final_fine_tuned_model.keras')
with open('model\tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

max_length = 35

# Function to map an index to a word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Function to generate caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

# Load VGG16 model and configure it to extract features
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Home route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and caption generation
@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)
        
        # Preprocess the image
        image = load_img(filepath, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        # Extract image features
        feature = vgg_model.predict(image, verbose=0)
        
        # Generate caption
        caption = predict_caption(model, feature, tokenizer, max_length)
        caption = caption.replace('startseq', '').replace('endseq', '')
        
        return render_template('index.html', caption=caption, image_url=filepath)

if __name__ == '__main__':
    app.run(debug=True)