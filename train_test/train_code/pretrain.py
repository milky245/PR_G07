import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import network
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--total_iter", default=50000, type=int)
    parser.add_argument("--adv_train_lr", default=2e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.75, type=float)
    parser.add_argument("--save_dir", default='pretrain')

    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_photo = torch.randn(args.batch_size, 3, args.patch_size, args.patch_size, device=device)
    model = network.UNetGenerator(channel=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.adv_train_lr, betas=(0.5, 0.99))

    face_photo_dir = '../../data/celeba-v2'
    face_photo_list = utils.load_image_list(face_photo_dir)
    scenery_photo_dir = '../../data/landscapes-v2'
    scenery_photo_list = utils.load_image_list(scenery_photo_dir)

    writer = SummaryWriter(log_dir=args.save_dir)

    for total_iter in tqdm(range(args.total_iter)):
        if total_iter % 5 == 0:
            photo_batch = utils.next_batch(face_photo_list, args.batch_size)
        else:
            photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)

        photo_batch = torch.tensor(photo_batch, dtype=torch.float32).to(device)

        optimizer.zero_grad()
        output = model(photo_batch)
        recon_loss = F.l1_loss(photo_batch, output)
        recon_loss.backward()
        optimizer.step()

        if (total_iter + 1) % 50 == 0:
            print(f'pretrain, iter: {total_iter + 1}, recon_loss: {recon_loss.item()}')
            writer.add_scalar('recon_loss', recon_loss.item(), total_iter)

            if (total_iter + 1) % 500 == 0:
                torch.save(model.state_dict(), f"{args.save_dir}/save_models/model_{total_iter + 1}.pth")

                photo_face = utils.next_batch(face_photo_list, args.batch_size)
                photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)

                with torch.no_grad():
                    result_face = model(torch.tensor(photo_face, dtype=torch.float32).to(device)).cpu().numpy()
                    result_scenery = model(torch.tensor(photo_scenery, dtype=torch.float32).to(device)).cpu().numpy()

                utils.write_batch_image(result_face, f"{args.save_dir}/images", f"{total_iter + 1}_face_result.jpg", 4)
                utils.write_batch_image(photo_face, f"{args.save_dir}/images", f"{total_iter + 1}_face_photo.jpg", 4)
                utils.write_batch_image(result_scenery, f"{args.save_dir}/images", f"{total_iter + 1}_scenery_result.jpg", 4)
                utils.write_batch_image(photo_scenery, f"{args.save_dir}/images", f"{total_iter + 1}_scenery_photo.jpg", 4)

if __name__ == '__main__':
    args = arg_parser()
    train(args)
