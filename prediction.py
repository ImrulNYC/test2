import os
import torch
from transformers import AutoConfig, AutoModelForImageClassification, ViTFeatureExtractor
from safetensors.torch import load_file
import urllib.request

# Bucket and model keys
BUCKET_URL = "https://flowerm.s3.amazonaws.com/"
MODEL_KEY = "model.safetensors"
CONFIG_KEY = "config.json"
PREPROCESSOR_KEY = "preprocessor_config.json"

# Temporary paths for downloaded files
model_path = "/tmp/model.safetensors"
config_path = "/tmp/config.json"
preprocessor_path = "/tmp/preprocessor_config.json"

# Function to download files from a public S3 URL
def download_file_from_s3(url, local_path):
    if not os.path.exists(local_path):
        try:
            urllib.request.urlretrieve(url, local_path)
            print(f"Downloaded {url} successfully.")
        except Exception as e:
            raise Exception(f"Failed to download {url}: {str(e)}")

# Cached loading of model components
@st.cache_resource
def load_model():
    # Download model files from S3
    download_file_from_s3(f"{BUCKET_URL}{MODEL_KEY}", model_path)
    download_file_from_s3(f"{BUCKET_URL}{CONFIG_KEY}", config_path)
    download_file_from_s3(f"{BUCKET_URL}{PREPROCESSOR_KEY}", preprocessor_path)

    # Load configuration, preprocessor, and model
    config = AutoConfig.from_pretrained(config_path)
    preprocessor = ViTFeatureExtractor.from_pretrained(preprocessor_path)
    state_dict = load_file(model_path)
    model = AutoModelForImageClassification.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=state_dict
    )

    # Label mappings
    id_to_label = {
        0: 'calendula', 1: 'coreopsis', 2: 'rose', 3: 'black_eyed_susan', 4: 'water_lily', 5: 'california_poppy',
        6: 'dandelion', 7: 'magnolia', 8: 'astilbe', 9: 'sunflower', 10: 'tulip', 11: 'bellflower',
        12: 'iris', 13: 'common_daisy', 14: 'daffodil', 15: 'carnation'
    }

    return model, preprocessor, id_to_label

# Function to make predictions
def predict_flower(image, model, preprocessor, id_to_label):
    inputs = preprocessor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = torch.max(probabilities).item() * 100
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = id_to_label.get(predicted_class, "Unknown")

    if confidence >= 80:
        return predicted_label, confidence
    else:
        return None, None
