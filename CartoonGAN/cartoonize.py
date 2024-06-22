import torch
import cv2
import numpy as np
from torchvision import transforms

def cartoonize(image_path):
    model = torch.jit.load("cartoon_gan.pt")
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32) / 127.5 - 1

    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
    output = (output.squeeze().numpy().transpose(1, 2, 0) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)

    cv2.imwrite('cartoon_image.jpg', output)

if __name__ == '__main__':
    image_path = 'test_image.jpg'
    cartoonize(image_path)
