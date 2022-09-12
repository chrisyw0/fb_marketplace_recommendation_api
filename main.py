import pickle
import requests
import os
import tensorflow as tf

from uuid import uuid4
from typing import Optional
from urllib.parse import unquote
from fastapi import FastAPI
from fbRecommendation.dl.tensorflow.tf_image_classifier import TFImageClassifier
from fbRecommendation.dl.tensorflow.tf_text_classifier_transformer import TFTextTransformerClassifier
from fbRecommendation.dl.tensorflow.tf_combine_classifier import TFImageTextClassifier
from fbRecommendation.dl.tensorflow.utils.tf_image_text_util import TFImageTextUtil

IMG_PATH = "./images/"

# load data and model
with open('./data/product_clean.pkl', 'rb') as f:
    df_product_clean = pickle.load(f)

with open('./data/image_clean.pkl', 'rb') as f:
    df_image_clean = pickle.load(f)

image_model = TFImageClassifier(df_product=df_product_clean, df_image=df_image_clean)
image_model.image_base_model = "EfficientNetB3"
image_model.load_model()

text_model = TFTextTransformerClassifier(df_product=df_product_clean, df_image=df_image_clean)
text_model.load_model()

combine_model = TFImageTextClassifier(
    df_product=df_product_clean,
    df_image=df_image_clean,
    image_seq_layers=image_model.image_seq_layers,
    text_seq_layers=text_model.text_seq_layer,
    embedding_model=text_model.embedding_model
)

combine_model.embedding = "BERT"
combine_model.load_model()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


def download_image(url: str, img_path: str, file_name: str = None) -> Optional[str]:
    print(f"Download image from URL {url}")
    local_filename = file_name if file_name else uuid4().hex + "." + url.split(".")[-1]
    os.makedirs(img_path, exist_ok=True)

    file_path = img_path + local_filename

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Image downloaded and saved to path {file_path}")

            return file_path

    except Exception as e:
        print(e)

        if os.path.exists(file_path):
            os.remove(file_path)

    return None


def get_image_tensor(file_path, resize=None):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.resize_with_pad(img, resize[0], resize[1])

    return img


@app.get("/predict")
@app.post("/predict")
async def predict(image_url: str = None, text: str = None):
    image_file = None
    text_tokens = None

    if image_url:
        image_file = download_image(
            unquote(image_url),
            img_path=IMG_PATH
        )

        if not image_file:
            return {
                "Error": "Unable to download image data"
            }

    if text:
        text_tokens = [TFImageTextUtil.clean_text(text)]

    input_data = None

    if image_file and text_tokens:
        this_model = combine_model
        image_data = [get_image_tensor(image_file, resize=this_model.image_shape)]

        input_data = {
            "image": tf.convert_to_tensor(image_data, dtype=tf.uint8),
            "token": tf.convert_to_tensor(text_tokens, dtype=tf.string)
        }

        os.remove(image_file)

    elif image_file:
        this_model = image_model
        input_data = tf.convert_to_tensor(
            [get_image_tensor(image_file, resize=this_model.image_shape)],
            dtype=tf.uint8
        )

    elif text_tokens:
        this_model = text_model
        input_data = text_tokens

    with tf.device('/CPU:0'):
        prediction = this_model.predict_model(input_data)[0]

    return {
        "Result": this_model.classes[prediction]
    }



