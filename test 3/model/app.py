from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from cxr_model import CXR_EfficientNetModel
import random

app = Flask(__name__, template_folder=r'C:\Users\Pushkraj\Desktop\test 3\templates', static_folder=r'C:\Users\Pushkraj\Desktop\test 3\static')

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
model = CXR_EfficientNetModel(num_classes=num_classes).to(device)

model.load_state_dict(torch.load(r"C:\Users\Pushkraj\Desktop\Trial v0\model\model (1).pth", map_location=device))

model.eval()

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Precautions
POSITIVE_PRECAUTIONS = [
    "Consult an oncologist for proper diagnosis and treatment plan.",
    "Avoid smoking and alcohol consumption.",
    "Maintain a healthy, balanced diet rich in fruits and vegetables.",
    "Get regular health check-ups and follow-ups.",
    "Stay physically active and manage your stress levels.",
    "Avoid exposure to harmful chemicals and radiation.",
    "Ensure adequate rest and sleep to strengthen immunity.",
    "Follow the prescribed medications and treatments properly."
]

NEGATIVE_PRECAUTIONS = [
    "Maintain a healthy lifestyle to prevent tumor development.",
    "Avoid exposure to environmental toxins and pollutants.",
    "Exercise regularly to improve overall health.",
    "Stay away from tobacco and alcohol.",
    "Get regular cancer screening tests.",
    "Stay informed about tumor prevention methods.",
    "Eat antioxidant-rich foods.",
    "Keep your immune system strong through proper rest and nutrition."
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part"
    file = request.files["file"]
    if file.filename == "":
        return "No selected file"
    if file:
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            prediction = "Tumor Detected" if predicted.item() == 1 else "Normal"

        precautions = POSITIVE_PRECAUTIONS if prediction == "Tumor Detected" else NEGATIVE_PRECAUTIONS
        random.shuffle(precautions)

        return render_template("result.html", prediction=prediction, precautions=precautions)

if __name__ == "__main__":
    app.run(debug=True)
