import torch
import open_clip
from PIL import Image
import torchvision.transforms as transforms

# Load CLIP model (same as training)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device)

# Load classifier head
classifier = torch.nn.Linear(model.visual.output_dim, 2).to(device)
classifier.load_state_dict(torch.load("website_aesthetic_classifier.pth", map_location=device))
classifier.eval()

def predict(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = model.encode_image(image)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
    
    classes = ['bad', 'good']
    predicted_class = classes[probs.argmax().item()]
    confidence = probs.max().item()

    return {"class": predicted_class, "confidence": confidence}
