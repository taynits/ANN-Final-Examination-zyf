import os
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import timm

# 加载模型与权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
model.head = nn.Sequential( model.head, nn.Linear(1000, 12) )
model = model.to(device)
model.load_state_dict(torch.load("results/model_Transformer_best.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# test
results = []
test_dir = './data/cat_12_test'

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()
    results.append((img_name, predicted_class))

output_file = 'results/Transformer_test_result.txt'
with open(output_file, 'w') as f:
    for img_name, predicted_class in results:
        f.write(f"{img_name} {predicted_class}\n")

print(f"Results written to {output_file}")