from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import torchvision.transforms as transforms

from src.model_hmt import HierarchicalMultimodalTransformerCLIP

app = Flask(__name__)
CORS(app)

# ---------------------
# Device
# ---------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------
# Load model
# ---------------------
model = HierarchicalMultimodalTransformerCLIP()
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ---------------------
# Image transform
# ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------
# API
# ---------------------
@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    text = request.form.get("text", "")

    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(
            text_raw=[text],
            pixel_values=img
        )

        prob = torch.sigmoid(output["sarcasm_presence"]).item()

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


@app.route("/")
def home():
    return "Sarcasm Detection API running 🚀"


if __name__ == "__main__":
    app.run()