import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import ToTensor, ToPILImage
import network
from guided_filter import guided_filter

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image

def cartoonize(load_folder, save_folder, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = network.UNetGenerator().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            batch_image = image.astype(np.float32) / 127.5 - 1
            batch_image = ToTensor()(batch_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(batch_image)
                output = guided_filter(batch_image, output, r=1, eps=5e-3)
            output = (output.squeeze().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except Exception as e:
            print(f'cartoonize {load_path} failed: {e}')

if __name__ == '__main__':
    model_path = 'saved_models/model.pth'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
