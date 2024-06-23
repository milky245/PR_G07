import torch
import torch.optim as optim
import numpy as np
import argparse
import os
import time
import cv2
from matplotlib import pyplot as plt

import model
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", default=96, type=int)
    parser.add_argument("--crop_size", default=16, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--pre_train_iter", default=2000, type=int)
    parser.add_argument("--iter", default=20000, type=int)
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

        self.is_train = torch.Tensor([True])
        self.photo_input = torch.Tensor(args.batch_size, 3, args.image_size, args.image_size)
        self.cartoon_input = torch.Tensor(args.batch_size, 3, args.image_size, args.image_size)
        self.blur_input = torch.Tensor(args.batch_size, 3, args.image_size, args.image_size)

    def input_setup(self):
        self.celeba_list = utils.get_filename_list('../data/celeba-v2')
        self.cartoon_list = utils.get_filename_list('../data/getchu-v2')
        print('Finished loading data')

    def build_model(self):
        self.generator = model.Generator().cuda()
        self.discriminator = model.MultiPatchDiscriminator(self.crop_size).cuda()
        self.criterion_vgg = utils.vgg_loss
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0., 0.9))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0., 0.9))
        print('Finished building model')

    def train(self):
        if not os.path.exists(self.train_out_dir):
            os.makedirs(self.train_out_dir)

        start_time = time.time()

        # 预训练迭代
        pretrain_path = os.path.join(self.save_dir, f'pre_train_generator_{self.pre_train_iter - 1}.pth')
        if os.path.isfile(pretrain_path):
            self.generator.load_state_dict(torch.load(pretrain_path))
            print('Finished loading pre_trained model')
        else:
            for iter in range(self.pre_train_iter):
                photo_batch = torch.Tensor(utils.next_batch(self.batch_size, self.image_size, self.celeba_list)).cuda()
                cartoon_batch, blur_batch = utils.next_blur_batch(self.batch_size, self.image_size, self.cartoon_list)
                cartoon_batch, blur_batch = torch.Tensor(cartoon_batch).cuda(), torch.Tensor(blur_batch).cuda()

                self.optimizer_g.zero_grad()
                fake_cartoon = self.generator(photo_batch)
                vgg_loss = self.criterion_vgg(photo_batch, fake_cartoon)
                g_loss = 5e3 * vgg_loss
                g_loss.backward()
                self.optimizer_g.step()

                if (iter + 1) % 50 == 0:
                    print('pre_train iteration:[%d/%d], vgg_loss:%f, time cost:%f' % (iter + 1, self.pre_train_iter, vgg_loss.item(), time.time() - start_time))
                    start_time = time.time()

                    if (iter + 1) % 500 == 0:
                        fake_cartoon = fake_cartoon.cpu().detach().numpy()
                        utils.print_fused_image(fake_cartoon, self.train_out_dir, str(iter) + '_pre_train.png', 4)

                    if (iter + 1) % self.pre_train_iter == 0:
                        torch.save(self.generator.state_dict(), os.path.join(self.save_dir, f'pre_train_generator_{iter}.pth'))

        # 训练迭代
        for iter in range(self.iter):
            photo_batch = torch.Tensor(utils.next_batch(self.batch_size, self.image_size, self.celeba_list)).cuda()
            cartoon_batch, blur_batch = utils.next_blur_batch(self.batch_size, self.image_size, self.cartoon_list)
            cartoon_batch, blur_batch = torch.Tensor(cartoon_batch).cuda(), torch.Tensor(blur_batch).cuda()

            for _ in range(3):
                self.optimizer_g.zero_grad()
                fake_cartoon = self.generator(photo_batch)
                vgg_loss = self.criterion_vgg(photo_batch, fake_cartoon)
                g_loss = -torch.mean(torch.log(torch.sigmoid(self.discriminator(fake_cartoon)))) + 5e3 * vgg_loss
                g_loss.backward()
                self.optimizer_g.step()

            self.optimizer_d.zero_grad()
            real_logit_cartoon = self.discriminator(cartoon_batch)
            fake_logit_cartoon = self.discriminator(fake_cartoon.detach())
            logit_blur = self.discriminator(blur_batch)
            d_loss = -torch.mean(torch.log(torch.sigmoid(real_logit_cartoon))
                                 + torch.log(1. - torch.sigmoid(fake_logit_cartoon))
                                 + torch.log(1. - torch.sigmoid(logit_blur)))
            d_loss.backward()
            self.optimizer_d.step()

            if (iter + 1) % 50 == 0:
                print('train iteration:[%d/%d], g_loss:%f, d_loss:%f, time cost:%f' % (iter + 1, self.iter, g_loss.item(), d_loss.item(), time.time() - start_time))
                start_time = time.time()

                if (iter + 1) % 500 == 0:
                    fake_cartoon = fake_cartoon.cpu().detach().numpy()
                    utils.print_fused_image(fake_cartoon, self.train_out_dir, str(iter) + '.png', 4)

                if (iter + 1) % 5000 == 0:
                    torch.save(self.generator.state_dict(), os.path.join(self.save_dir, 'generator_%d.pth' % iter))
                    torch.save(self.discriminator.state_dict(), os.path.join(self.save_dir, 'discriminator_%d.pth' % iter))

    def test(self):
        if not os.path.exists(self.test_out_dir):
            os.mkdir(self.test_out_dir)
        self.test_list = utils.get_filename_list('../data/test-v2')
        self.generator.load_state_dict(torch.load(os.path.join(self.save_dir, 'generator_%d.pth' % self.iter)))
        self.generator.eval()
        for idx in range(100):
            photo_batch = torch.Tensor(utils.next_batch_no_resize(self.batch_size, self.image_size, self.test_list)).cuda()
            with torch.no_grad():
                fake_cartoon = self.generator(photo_batch)
            fake_cartoon = fake_cartoon.cpu().numpy()
            utils.print_fused_image(fake_cartoon, self.test_out_dir, str(idx) + '.png', 4)
            utils.print_fused_single_image(fake_cartoon[0], self.test_out_dir, f'test_image_{idx}.png')

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

# import torch
# import torch.optim as optim
# import numpy as np
# import argparse
# import os
# import time
# import cv2
# from matplotlib import pyplot as plt

# import model
# import utils

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# def arg_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_size", default=96, type=int)
#     parser.add_argument("--crop_size", default=16, type=int)
#     parser.add_argument("--batch_size", default=16, type=int)
#     parser.add_argument("--pre_train_iter", default=2000, type=int)
#     parser.add_argument("--iter", default=15000, type=int)
#     parser.add_argument("--learning_rate", default=1e-4, type=float)
#     parser.add_argument("--gpu_fraction", default=0.75, type=float)
#     parser.add_argument("--save_dir", default='saved_models')
#     parser.add_argument("--train_out_dir", default='train_output')
#     parser.add_argument("--test_out_dir", default='test_output')
#     parser.add_argument("--mode", default='train')
#     args = parser.parse_args()
#     return args

# class CartoonGAN:
#     def __init__(self, args):
#         self.image_size = args.image_size
#         self.crop_size = args.crop_size
#         self.batch_size = args.batch_size
#         self.pre_train_iter = args.pre_train_iter
#         self.iter = args.iter
#         self.learning_rate = args.learning_rate
#         self.gpu_fraction = args.gpu_fraction
#         self.train_out_dir = args.train_out_dir
#         self.test_out_dir = args.test_out_dir
#         self.save_dir = args.save_dir
#         self.lambda_ = 10

#         self.is_train = torch.Tensor([True])
#         self.photo_input = torch.Tensor(args.batch_size, 3, args.image_size, args.image_size)
#         self.cartoon_input = torch.Tensor(args.batch_size, 3, args.image_size, args.image_size)
#         self.blur_input = torch.Tensor(args.batch_size, 3, args.image_size, args.image_size)

#     def input_setup(self):
#         self.celeba_list = utils.get_filename_list('../data/celeba-v2')
#         self.cartoon_list = utils.get_filename_list('../data/getchu-v2')
#         print('Finished loading data')

#     def build_model(self):
#         self.generator = model.Generator().cuda()
#         self.discriminator = model.MultiPatchDiscriminator(self.crop_size).cuda()
#         self.criterion_vgg = utils.vgg_loss
#         self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0., 0.9))
#         self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0., 0.9))
#         print('Finished building model')

#     def train(self):
#         if not os.path.exists(self.train_out_dir):
#             os.makedirs(self.train_out_dir)
#         for iter in range(self.iter):
#             photo_batch = torch.Tensor(utils.next_batch(self.batch_size, self.image_size, self.celeba_list)).cuda()
#             cartoon_batch, blur_batch = utils.next_blur_batch(self.batch_size, self.image_size, self.cartoon_list)
#             cartoon_batch, blur_batch = torch.Tensor(cartoon_batch).cuda(), torch.Tensor(blur_batch).cuda()

#             self.optimizer_g.zero_grad()
#             fake_cartoon = self.generator(photo_batch)
#             vgg_loss = self.criterion_vgg(photo_batch, fake_cartoon)
#             g_loss = -torch.mean(torch.log(torch.sigmoid(self.discriminator(fake_cartoon)))) + 5e3 * vgg_loss
#             g_loss.backward()
#             self.optimizer_g.step()

#             self.optimizer_d.zero_grad()
#             real_logit_cartoon = self.discriminator(cartoon_batch)
#             fake_logit_cartoon = self.discriminator(fake_cartoon.detach())
#             logit_blur = self.discriminator(blur_batch)
#             d_loss = -torch.mean(torch.log(torch.sigmoid(real_logit_cartoon))
#                                  + torch.log(1. - torch.sigmoid(fake_logit_cartoon))
#                                  + torch.log(1. - torch.sigmoid(logit_blur)))
#             d_loss.backward()
#             self.optimizer_d.step()

#             if (iter+1) % 50 == 0:
#                 print('train iteration:[%d/%d], g_loss:%f, d_loss:%f' % (iter + 1, self.iter, g_loss.item(), d_loss.item()))

#                 if (iter+1) % 500 == 0:
#                     fake_cartoon = fake_cartoon.cpu().detach().numpy()
#                     utils.print_fused_image(fake_cartoon, self.train_out_dir, str(iter) + '.png', 4)

#                 if (iter+1) % 5000 == 0:
#                     torch.save(self.generator.state_dict(), os.path.join(self.save_dir, 'generator_%d.pth' % iter))
#                     torch.save(self.discriminator.state_dict(), os.path.join(self.save_dir, 'discriminator_%d.pth' % iter))

#     def test(self):
#         if not os.path.exists(self.test_out_dir):
#             os.mkdir(self.test_out_dir)
#         self.test_list = utils.get_filename_list('../data/test-v2')
#         self.generator.load_state_dict(torch.load(os.path.join(self.save_dir, 'generator_%d.pth' % self.iter)))
#         self.generator.eval()
#         for idx in range(100):
#             photo_batch = torch.Tensor(utils.next_batch(self.batch_size, self.image_size, self.test_list)).cuda()
#             with torch.no_grad():
#                 fake_cartoon = self.generator(photo_batch)
#             fake_cartoon = fake_cartoon.cpu().numpy()
#             utils.print_fused_image(fake_cartoon, self.test_out_dir, str(idx) + '.png', 4)

# def main():
#     args = arg_parser()
#     model = CartoonGAN(args)
#     if args.mode == 'train':
#         model.build_model()
#         model.input_setup()
#         model.train()
#     elif args.mode == 'test':
#         model.build_model()
#         model.test()

# if __name__ == '__main__':
#     main()
