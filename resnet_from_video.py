import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import cv2

checkpoint = torch.load('./models/ResNet18.pth', map_location='cpu')  # или 'cuda' при наличии
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = transform(frame)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

    cv2.putText(frame, f'Predicted: {class_names[predicted_class]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


