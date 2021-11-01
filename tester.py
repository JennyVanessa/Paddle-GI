import os
#import torch
#import torch.nn as nn
#from torch import autograd

import paddle
import paddle.nn as nn
from paddle import autograd

from model.networks import Generator, LocalDis, GlobalDis

#from duiqi import transfer


from utils.tools import get_model_list, local_patch, spatial_discounting_mask
from utils.logger import get_logger
import numpy as np

logger = get_logger()


class Trainer(nn.Layer):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)
        # transfer(self.globalD)

        # paddle_checkpoint = paddle.load('gen_00042500.pdparams')
        # self.netG.set_state_dict(paddle_checkpoint)
        # paddle_checkpoint = paddle.load('paddle2.pdparams')
        # self.localD.set_state_dict(paddle_checkpoint)
        # paddle_checkpoint = paddle.load('paddle3.pdparams')
        # self.globalD.set_state_dict(paddle_checkpoint)

        # self.optimizer_g = paddle.optimizer.Adam(parameters=self.netG.parameters(), learning_rate=self.config['lr'],
        #                                     beta1=self.config['beta1'], beta2=self.config['beta2'])
        # d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        # self.optimizer_d = paddle.optimizer.Adam(parameters=d_params, learning_rate=config['lr'],
        #                                     beta1=self.config['beta1'], beta2=self.config['beta2'])

        self.optimizer_g = paddle.optimizer.Adam(parameters=self.netG.parameters(), learning_rate=self.config['lr'],
                                            beta1=self.config['beta1'], beta2=self.config['beta2'],grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0))
        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        self.optimizer_d = paddle.optimizer.Adam(parameters=d_params, learning_rate=config['lr'],
                                            beta1=self.config['beta1'], beta2=self.config['beta2'],grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0))

        # if self.use_cuda:
        #     self.netG.to(self.device_ids[0])
        #     self.localD.to(self.device_ids[0])
        #     self.globalD.to(self.device_ids[0])

    def forward(self, x, bboxes, masks, ground_truth, compute_loss_g=False):
        self.train()
        #l1_loss = nn.L1Loss()
        #losses = {}
        x1, x2, offset_flow = self.netG(x, masks)
        #local_patch_gt = local_patch(ground_truth, bboxes)
        #x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)
        # local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        # local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes)
        # #print(paddle.max(x1_inpaint-ground_truth))

        # # D part
        # # wgan d loss
        # local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
        #     self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        # global_real_pred, global_fake_pred = self.dis_forward(
        #     self.globalD, ground_truth, x2_inpaint.detach())
        # losses['wgan_d'] = paddle.mean(local_patch_fake_pred - local_patch_real_pred) + \
        #    paddle.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # # losses['wgan_d'] = paddle.abs(paddle.mean(local_patch_real_pred - local_patch_fake_pred )) + \
        # #      paddle.abs(paddle.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha'])
        # # gradients penalty loss
        # local_penalty = self.calc_gradient_penalty(
        #     self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        # global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
        # losses['wgan_gp'] = local_penalty + global_penalty

        # # G part
        # if compute_loss_g:
        #     sd_mask = spatial_discounting_mask(self.config)
        #     #print(np.max((local_patch_x1_inpaint * sd_mask-local_patch_gt * sd_mask).numpy()))
        #     losses['l1'] = l1_loss(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) * \
        #         self.config['coarse_l1_alpha'] + \
        #         l1_loss(local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask)
        #     losses['ae'] = l1_loss(x1 * (1. - masks), ground_truth * (1. - masks)) * \
        #         self.config['coarse_l1_alpha'] + \
        #         l1_loss(x2 * (1. - masks), ground_truth * (1. - masks))

        #     # wgan g loss
        #     local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
        #         self.localD, local_patch_gt, local_patch_x2_inpaint)
        #     global_real_pred, global_fake_pred = self.dis_forward(
        #         self.globalD, ground_truth, x2_inpaint)
        #     losses['wgan_g'] = - paddle.mean(local_patch_fake_pred) - \
        #         paddle.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

        #return losses, x2_inpaint, offset_flow

        return x2_inpaint

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.shape == x_inpaint.shape
        #batch_size = ground_truth.shape[0]
        batch_data = paddle.concat([ground_truth, x_inpaint], axis=0)
        batch_output = netD(batch_data)
        real_pred,fake_pred=paddle.split(batch_output,2,axis=0)

        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.shape[0]
        alpha = paddle.rand(shape=[batch_size, 1, 1, 1])
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        #interpolates = interpolates.requires_grad_().clone()
        interpolates = paddle.to_tensor(interpolates,stop_gradient=False)

        disc_interpolates = netD(interpolates)
        grad_outputs = paddle.ones(disc_interpolates.shape)

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        #gradients = gradients.view(batch_size, -1)
        gradients = paddle.reshape(gradients,[batch_size,-1])
        #gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_penalty = paddle.mean((paddle.norm(gradients,p=2,axis=1)-1)**2)

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks)
        # x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        return x2_inpaint, offset_flow

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pdparams' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pdparams' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pdparams')
        paddle.save(self.netG.state_dict(), gen_name)
        paddle.save({'localD': self.localD.state_dict(),
                    'globalD': self.globalD.state_dict()}, dis_name)
        paddle.save({'gen': self.optimizer_g.state_dict(),
                    'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        #last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        #self.netG.set_state_dict(paddle.load(last_model_name))
        self.netG.set_state_dict(paddle.load(checkpoint_dir+"gen_00141000.pdparams"))
        #iteration = int(last_model_name[-14:-9])
        iteration = 141000
        if not test:
            # Load discriminators
            #last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            last_model_name = checkpoint_dir+"dis_00141000.pdparams"
            state_dict = paddle.load(last_model_name)
            self.localD.set_state_dict(state_dict['localD'])
            self.globalD.set_state_dict(state_dict['globalD'])
            # Load optimizers
            state_dict = paddle.load(os.path.join(checkpoint_dir, 'optimizer.pdparams'))
            self.optimizer_d.set_state_dict(state_dict['dis'])
            self.optimizer_g.set_state_dict(state_dict['gen'])

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration
