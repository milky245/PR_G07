import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import loss
import network
import utils
from guided_filter import guided_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--total_iter", default=100000, type=int)
    parser.add_argument("--adv_train_lr", default=2e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.75, type=float)
    parser.add_argument("--save_dir", default='train_cartoon', type=str)
    parser.add_argument("--use_enhance", default=False)

    args = parser.parse_args()
    return args

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_photo = torch.randn(args.batch_size, 3, args.patch_size, args.patch_size, device=device)
    input_superpixel = torch.randn(args.batch_size, 3, args.patch_size, args.patch_size, device=device)
    input_cartoon = torch.randn(args.batch_size, 3, args.patch_size, args.patch_size, device=device)

    model = network.UNetGenerator().to(device)
    optimizer_g = optim.Adam(model.parameters(), lr=args.adv_train_lr, betas=(0.5, 0.99))
    optimizer_d = optim.Adam(network.Discriminator().parameters(), lr=args.adv_train_lr, betas=(0.5, 0.99))

    writer = SummaryWriter(log_dir=args.save_dir)

    face_photo_dir = '../../data/celeba-v2'
    face_photo_list = utils.load_image_list(face_photo_dir)
    scenery_photo_dir = '../../data/landscapes-v2'
    scenery_photo_list = utils.load_image_list(scenery_photo_dir)

    face_cartoon_dir = '../../data/getchu-v2'
    face_cartoon_list = utils.load_image_list(face_cartoon_dir)
    scenery_cartoon_dir = '../../data/cartoons-v2'
    scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir)

    for total_iter in tqdm(range(args.total_iter)):
        if total_iter % 5 == 0:
            photo_batch = utils.next_batch(face_photo_list, args.batch_size)
            cartoon_batch = utils.next_batch(face_cartoon_list, args.batch_size)
        else:
            photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
            cartoon_batch = utils.next_batch(scenery_cartoon_list, args.batch_size)

        photo_batch = torch.tensor(photo_batch, dtype=torch.float32).to(device)
        cartoon_batch = torch.tensor(cartoon_batch, dtype=torch.float32).to(device)

        inter_out = model(photo_batch)
        if args.use_enhance:
            superpixel_batch = utils.selective_adacolor(inter_out.cpu().numpy(), power=1.2)
        else:
            superpixel_batch = utils.simple_superpixel(inter_out.cpu().numpy(), seg_num=200)
        superpixel_batch = torch.tensor(superpixel_batch, dtype=torch.float32).to(device)

        optimizer_g.zero_grad()
        output = model(photo_batch)
        recon_loss = loss.recon_loss(output, photo_batch, superpixel_batch)
        g_loss = loss.generator_loss(output, cartoon_batch)
        g_loss_total = recon_loss + g_loss
        g_loss_total.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        d_loss = loss.discriminator_loss(cartoon_batch, output)
        d_loss.backward()
        optimizer_d.step()

        if (total_iter + 1) % 50 == 0:
            print(f'Iter: {total_iter + 1}, d_loss: {d_loss.item()}, g_loss: {g_loss_total.item()}, recon_loss: {recon_loss.item()}')
            writer.add_scalar('d_loss', d_loss.item(), total_iter)
            writer.add_scalar('g_loss', g_loss_total.item(), total_iter)
            writer.add_scalar('recon_loss', recon_loss.item(), total_iter)

            if (total_iter + 1) % 500 == 0:
                torch.save(model.state_dict(), f"{args.save_dir}/saved_models/model_{total_iter + 1}.pth")

                photo_face = utils.next_batch(face_photo_list, args.batch_size)
                cartoon_face = utils.next_batch(face_cartoon_list, args.batch_size)
                photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)
                cartoon_scenery = utils.next_batch(scenery_cartoon_list, args.batch_size)

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
