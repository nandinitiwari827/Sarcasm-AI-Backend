from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import gdown
import requests
from io import BytesIO

from src.model_hmt import HierarchicalMultimodalTransformerCLIP

app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🖥 Device:", device)

# -------------------------------------------------
# MODEL DOWNLOAD
# -------------------------------------------------
MODEL_PATH = "best_model.pt"
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1Q_Oh_hxjwMGK8e5HZzJtM7UFu-S2P8kk"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading AI model from Google Drive...")
    gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)
else:
    print("✅ Model already exists")

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
print("⏳ Loading model...")

model = HierarchicalMultimodalTransformerCLIP()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# -------------------------------------------------
# IMAGE TRANSFORM
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------------------------
# ROUTES
# -------------------------------------------------

@app.route("/")
def home():
    return "Sarcasm Detection AI Backend Running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = request.files.get("image")
        image_url = request.form.get("image_url", "")
        text = request.form.get("text", "")

        # ------------------------------
        # IMAGE FROM DEVICE
        # ------------------------------
        if image:
            img = Image.open(image).convert("RGB")

        # ------------------------------
        # IMAGE FROM URL (IMPORTANT FIX)
        # ------------------------------
        elif image_url:
            response = requests.get(image_url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")

        else:
            return jsonify({"error": "No image provided"}), 400

        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(
                text_raw=[text],
                pixel_values=img
            )

            prob = torch.sigmoid(output["sarcasm_presence"]).item()

        # Human-friendly output
        if prob < 0.2:
            label = "Not Sarcastic 🙂"
        elif prob < 0.5:
            label = "Mild Sarcasm 😐"
        elif prob < 0.8:
            label = "Sarcastic 😏"
        else:
            label = "Highly Sarcastic 🔥"

        return jsonify({
            "sarcasm_probability": round(prob, 3),
            "result": label
        })

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": "Prediction failed"}), 500


# -------------------------------------------------
# START SERVER (HF / Render)
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)