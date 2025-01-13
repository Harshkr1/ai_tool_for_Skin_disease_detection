import torch
from torchvision import transforms
from PIL import Image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

# Define the PretrainedEfficientNet class
class PretrainedEfficientNet(nn.Module):
    def __init__(self, num_class=7):  # Adjust `num_class` based on your dataset
        super(PretrainedEfficientNet, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')  # Use the correct EfficientNet version
        self.fc1 = nn.Linear(self.efficientnet._fc.in_features, num_class)

    def forward(self, x):
        x = self.efficientnet.extract_features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "diagnosis/model/best_model.pth"  # Update path as needed
model = PretrainedEfficientNet(num_class=7)
model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
model.eval()
model.to(device)

# Define image transformations
input_size = 224
norm_mean = [0.7630362, 0.5456468, 0.5700442]
norm_std = [0.14092818, 0.15261286, 0.16997077]
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])

# Lesion type dictionary
lesion_type_dict = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis-like lesions',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic nevi',
    'vasc': 'Vascular lesions',
    'mel': 'Melanoma'
}

# Prediction function
def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    
    # Map prediction to class name
    predicted_class_idx = predicted_class.item()
    predicted_class_name = list(lesion_type_dict.values())[predicted_class_idx]
    
    return predicted_class_name

# Django view for uploading and predicting
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        # Handle uploaded file
        image = request.FILES['image']
        fs = FileSystemStorage()
        file_path = fs.save(image.name, image)
        full_path = fs.path(file_path)
        
        # Perform prediction
        predicted_class_name = predict_image(full_path)
        
        return render(request, 'result.html', {
            'predicted_class': predicted_class_name,
            'image_url': fs.url(file_path)
        })
    return render(request, 'upload.html')
