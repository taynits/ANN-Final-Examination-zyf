import os
from PIL import Image
import torch
from torchvision import transforms
from models.CNN import ResNet, Bottleneck

# 加载模型与权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(Bottleneck, [3, 4, 6, 3])
print(model)
model = model.to(device)
model.load_state_dict(torch.load("results/model_CNN_best.pth"))
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

output_file = 'results/CNN_test_result.txt'
with open(output_file, 'w') as f:
    for img_name, predicted_class in results:
        f.write(f"{img_name} {predicted_class}\n")

print(f"Results written to {output_file}")

