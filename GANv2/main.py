import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import os
import time
import cv2
import utils
from model import Generator, MultiPatchDiscriminator

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default=96, type=int)
    parser.add_argument("--crop_size", default=16, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--pre_train_iter", default=20000, type=int)
    parser.add_argument("--iter", default=100000, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.75, type=float)
    parser.add_argument("--save_dir", default='saved_models')
    parser.add_argument("--train_out_dir", default='train_output')
    parser.add_argument("--test_out_dir", default='test_output')
    parser.add_argument("--mode", default='train')

    args = parser.parse_args()
    return args

class CartoonGAN:
    def __init__(self, args):
        self.image_size = args.image_size
        self.crop_size = args.crop_size
        self.batch_size = args.batch_size
        self.pre_train_iter = args.pre_train_iter
        self.iter = args.iter
        self.learning_rate = args.learning_rate
        self.gpu_fraction = args.gpu_fraction
        self.train_out_dir = args.train_out_dir
        self.test_out_dir = args.test_out_dir
        self.save_dir = args.save_dir
        self.lambda_ = 10

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = Generator().to(self.device)
        self.discriminator = MultiPatchDiscriminator(self.crop_size).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0., 0.9))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0., 0.9))

        self.writer = SummaryWriter(log_dir=self.save_dir)

    def input_setup(self):
        self.celeba_list = utils.get_filename_list('../data/celeba-v2')
        self.cartoon_list = utils.get_filename_list('../data/getchu-v2')
        print('Finished loading data')

    def vgg_loss(self, image_a, image_b):
        return utils.vgg_loss(image_a, image_b)

    def build_model(self):
        print('Model built')

    def train(self):
        if not os.path.exists(self.train_out_dir):
            os.makedirs(self.train_out_dir)

        start_time = time.time()

        for iter in range(self.pre_train_iter):
            photo_batch = utils.next_batch(self.batch_size, self.image_size, self.celeba_list)
            cartoon_batch, blur_batch = utils.next_blur_batch(self.batch_size, self.image_size, self.cartoon_list)

            photo_batch = torch.tensor(photo_batch).permute(0, 3, 1, 2).to(self.device).float()
            cartoon_batch = torch.tensor(cartoon_batch).permute(0, 3, 1, 2).to(self.device).float()
            blur_batch = torch.tensor(blur_batch).permute(0, 3, 1, 2).to(self.device).float()

            self.optimizer_g.zero_grad()
            fake_cartoon = self.generator(photo_batch)
            VGG_loss = self.vgg_loss(photo_batch, fake_cartoon)
            VGG_loss.backward()
            self.optimizer_g.step()

            if (iter + 1) % 50 == 0:
                print(f'pre_train iteration:[{iter + 1}/{self.pre_train_iter}], time cost:{time.time() - start_time:.4f}')
                start_time = time.time()

                if (iter + 1) % 1000 == 0:
                    utils.print_fused_image(fake_cartoon.cpu().detach().numpy(), self.train_out_dir, f'{iter}_pre_train.png', 4)

                if (iter + 1) == self.pre_train_iter:
                    torch.save(self.generator.state_dict(), os.path.join(self.save_dir, 'pre_train.pth'))

        for iter in range(self.iter):
            photo_batch = utils.next_batch(self.batch_size, self.image_size, self.celeba_list)
            cartoon_batch, blur_batch = utils.next_blur_batch(self.batch_size, self.image_size, self.cartoon_list)

            photo_batch = torch.tensor(photo_batch).permute(0, 3, 1, 2).to(self.device).float()
            cartoon_batch = torch.tensor(cartoon_batch).permute(0, 3, 1, 2).to(self.device).float()
            blur_batch = torch.tensor(blur_batch).permute(0, 3, 1, 2).to(self.device).float()

            # Train generator
            self.optimizer_g.zero_grad()
            fake_cartoon = self.generator(photo_batch)
            VGG_loss = self.vgg_loss(photo_batch, fake_cartoon)
            g_loss = -torch.mean(torch.sigmoid(self.discriminator(fake_cartoon))) + 5e3 * VGG_loss
            g_loss.backward()
            self.optimizer_g.step()

            # Train discriminator
            self.optimizer_d.zero_grad()
            real_logit_cartoon = self.discriminator(cartoon_batch)
            fake_logit_cartoon = self.discriminator(fake_cartoon.detach())
            logit_blur = self.discriminator(blur_batch)
            d_loss = -torch.mean(torch.sigmoid(real_logit_cartoon)) + \
                     torch.mean(torch.sigmoid(fake_logit_cartoon)) + \
                     torch.mean(torch.sigmoid(logit_blur))
            d_loss.backward()
            self.optimizer_d.step()

            self.writer.add_scalar('g_loss', g_loss.item(), iter)
            self.writer.add_scalar('d_loss', d_loss.item(), iter)

            if (iter + 1) % 10 == 0:
                print(f'train iteration:[{iter + 1}/{self.iter}], time cost:{time.time() - start_time:.4f}')
                start_time = time.time()

                if (iter + 1) % 500 == 0:
                    utils.print_fused_image(fake_cartoon.cpu().detach().numpy(), self.train_out_dir, f'{iter}.png', 4)

                if (iter + 1) % 20000 == 0:
                    torch.save(self.generator.state_dict(), os.path.join(self.save_dir, f'model_{iter}.pth'))

    def test(self):
        if not os.path.exists(self.test_out_dir):
            os.makedirs(self.test_out_dir)

        self.generator.load_state_dict(torch.load(os.path.join(self.save_dir, 'model.pth')))
        self.generator.eval()

        test_list = utils.get_filename_list('../data/test-v2')

        for idx in range(100):
            photo_batch = utils.next_batch(self.batch_size, self.image_size, test_list)
            photo_batch = torch.tensor(photo_batch).permute(0, 3, 1, 2).to(self.device).float()
            with torch.no_grad():
                images = self.generator(photo_batch)
            utils.print_fused_image(images.cpu().numpy(), self.test_out_dir, f'{idx}.png', 4)

def main():
    args = arg_parser()
    model = CartoonGAN(args)

    if args.mode == 'train':
        model.build_model()
        model.input_setup()
        model.train()

    elif args.mode == 'test':
        model.build_model()
        model.test()

if __name__ == '__main__':
    main()
