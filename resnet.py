import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import cv2
import os
from collections import Counter
import time

checkpoint = torch.load('./models/ResNet18.pth', map_location='cpu')
model_weights = checkpoint['MODEL_STATE']

model = resnet18(pretrained=False, num_classes=34)
model.load_state_dict(model_weights)
model.eval()

class_names = [
    "grabbing", "grip", "holy", "point", "call", "three3", "timeout", "xsign",
    "hand_heart", "hand_heart2", "little_finger", "middle_finger", "take_picture",
    "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace",
    "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up",
    "two_up_inverted", "three_gun", "thumb_index", "thumb_index2", "no_gesture"
]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image_folder = './data/stop'
gesture_counts = Counter()

start_time = time.time()

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Ошибка чтения файла: {img_path}")
            continue

        input_tensor = transform(img)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = model(input_batch)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
            gesture_counts[class_names[predicted_class]] += 1

end_time = time.time()

print("Распознанные жесты:")
for gesture, count in gesture_counts.items():
    print(f"{gesture}: {count}")

elapsed_time = end_time - start_time
print(f"\nОбщее время работы алгоритма: {elapsed_time:.2f} секунд")
