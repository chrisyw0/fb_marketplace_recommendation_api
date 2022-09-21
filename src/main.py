import re
import emoji
import requests
import os
import tensorflow as tf
import tensorflow_text as text  # Registers the ops.
import numpy as np

from uuid import uuid4
from typing import Optional
from urllib.parse import unquote
from fastapi import FastAPI
from fbRecommendation.dl.tensorflow.model.tf_model import \
    TFImageModel, TFTextTransformerModel, TFCombineModel


IMG_PATH = "../images/"
image_shape = (300, 300, 3)

kwargs = dict(
    num_class=13,
    model_name="image_model_efficientNet",
    dropout_conv=0.6,
    dropout_pred=0.4,
    image_shape=image_shape,
    image_base_model="EfficientNetB3"
)
image_model, _, image_seq_layer = TFImageModel.get_model(**kwargs)
TFImageModel.load_model(image_model, "./model/tf_image_model_EfficientNetB3/weights/model.ckpt")

kwargs = dict(
    num_class=13,
    model_name="text_model_BERT",
    embedding="BERT",
    embedding_dim=768,
    embedding_pretrain_model="bert_en_cased_L-12_H-768_A-12",
    dropout_pred=0.5
)

text_model, text_seq_layer, _ = TFTextTransformerModel.get_model(**kwargs)
TFTextTransformerModel.load_model(text_model, "./model/tf_text_model_BERT/weights/model.ckpt")

kwargs = dict(
    num_class=13,
    model_name="combine_model_efficientNet_BERT",
    is_transformer_based_text_model=True,
    num_max_tokens=-1,
    image_shape=image_shape,
    text_seq_layers=tf.keras.models.clone_model(text_seq_layer),
    image_seq_layers=tf.keras.models.clone_model(image_seq_layer),
    image_base_model="EfficientNetB3"
)

combine_model = TFCombineModel.get_model(**kwargs)
TFCombineModel.load_model(combine_model, "./model/tf_image_text_model_EfficientNetB3_BERT/weights/model.ckpt")

classes = [
    "Appliances", "Baby & Kids Stuff",
    "Clothes, Footwear & Accessories",
    "Computers & Software",
    "DIY Tools & Materials",
    "Health & Beauty",
    "Home & Garden",
    "Music, Films, Books & Games",
    "Office Furniture & Equipment",
    "Other Goods",
    "Phones, Mobile Phones & Telecoms",
    "Sports, Leisure & Travel",
    "Video Games & Consoles"
]

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


def clean_text(text):
    result_text = emoji.replace_emoji(text, replace='')
    result_text = re.sub('[\n\t\r|]', '', result_text)
    result_text = re.sub(' +', ' ', result_text)

    result_text = result_text.strip()

    return result_text


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
        text_tokens = [clean_text(text)]

    input_data = None

    if image_file and text_tokens:
        this_model = combine_model
        image_data = [get_image_tensor(image_file, resize=image_shape)]

        input_data = {
            "image": tf.convert_to_tensor(image_data, dtype=tf.uint8),
            "token": tf.convert_to_tensor(text_tokens, dtype=tf.string)
        }

        os.remove(image_file)

    elif image_file:
        this_model = image_model
        input_data = tf.convert_to_tensor(
            [get_image_tensor(image_file, resize=image_shape)],
            dtype=tf.uint8
        )

        os.remove(image_file)

    elif text_tokens:
        this_model = text_model
        input_data = text_tokens

    with tf.device('/CPU:0'):
        prediction = np.argmax(this_model.predict(input_data)[0])

    return {
        "Result": classes[prediction]
    }



