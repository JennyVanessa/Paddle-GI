import os
import random
import time
import shutil
from argparse import ArgumentParser
#from paddle.nn.layer import loss

#import torch
#import torch.nn as nn
#import torch.backends.cudnn as cudnn
#import torchvision.utils as vutils

import paddle
import paddle.nn as nn
import paddle.vision
import numpy as np

#from tensorboardX import SummaryWriter

from trainer import Trainer
from data.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image
from utils.logger import get_logger
#from reprod_log import ReprodLogger
import scipy.misc
import cv2


#random.seed(123)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        #cudnn.benchmark = True

    # Configure checkpoint path
    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
    #writer = SummaryWriter(logdir=checkpoint_path)
    logger = get_logger(checkpoint_path)    # get logger and configure it at the first call

    logger.info("Arguments: {}".format(args))
    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    paddle.seed(args.seed)
    #if cuda:
    #    paddle.cuda.manual_seed_all(args.seed)

    # Log the configuration
    logger.info("Configuration: {}".format(config))

    #try:  # for unexpected error logging
        # Load the dataset
    logger.info("Training on dataset: {}".format(config['dataset_name']))
    train_dataset = Dataset(data_path=config['train_data_path'],
                            with_subfolder=config['data_with_subfolder'],
                            image_shape=config['image_shape'],
                            random_crop=config['random_crop'])
    # val_dataset = Dataset(data_path=config['val_data_path'],
    #                       with_subfolder=config['data_with_subfolder'],
    #                       image_size=config['image_size'],
    #                       random_crop=config['random_crop'])
    train_loader = paddle.io.DataLoader(dataset=train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=True,
                                                num_workers=config['num_workers'],
                                                use_shared_memory=True,
                                                use_buffer_reader=True)
    # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
    #                                           batch_size=config['batch_size'],
    #                                           shuffle=False,
    #                                           num_workers=config['num_workers'])

    # Define the trainer
    trainer = Trainer(config)
    logger.info("\n{}".format(trainer.netG))
    logger.info("\n{}".format(trainer.localD))
    logger.info("\n{}".format(trainer.globalD))

    if cuda:
        #trainer = paddle.DataParallel(trainer, device_ids=device_ids)
        #trainer = paddle.DataParallel(trainer)
        trainer_module = trainer#.module
    else:
        trainer_module = trainer

    # Get the resume iteration to restart training
    start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1

    iterable_train_loader = iter(train_loader)

    time_count = time.time()


    for iteration in range(start_iteration, config['niter'] + 1):

        try:
            ground_truth = next(iterable_train_loader)
            ground_truth = paddle.to_tensor(ground_truth)
        except StopIteration:
            iterable_train_loader = iter(train_loader)
            ground_truth = next(iterable_train_loader)
            ground_truth = paddle.to_tensor(ground_truth)

        #print(ground_truth)

        # Prepare the inputs
        bboxes = random_bbox(config, batch_size=ground_truth.shape[0])
        x, mask = mask_image(ground_truth, bboxes, config)
        #if cuda:
        #    x = x.cuda()
        #    mask = mask.cuda()
        #    ground_truth = ground_truth.cuda()

        #for i in range(2):

        ###### Forward pass ######
        compute_g_loss = iteration % config['n_critic'] == 0
        losses, inpainted_result, offset_flow = trainer(x, bboxes, mask, ground_truth, compute_g_loss)

        # Scalars from different devices are gathered into vectors
        for k in losses.keys():
            if not losses[k].dim() == 0:
                losses[k] = paddle.mean(losses[k])



        # if iteration==1:
        #     print(losses)
        #     return losses

        # print(losses)
        # print(losses["wgan_d"]+losses["wgan_gp"])
        # reprod_logger = ReprodLogger()
        # reprod_logger.add("logits", (losses["wgan_d"]+losses["wgan_gp"]).cpu().detach().numpy())
        # reprod_logger.save("loss_paddle.npy")

        ###### Backward pass ######
        # Update D
        trainer_module.optimizer_d.clear_grad()
        losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
        if paddle.abs(losses['d']).item()<30:
            losses['d'].backward()
            trainer_module.optimizer_d.step()

        # Update G
        if compute_g_loss:
            trainer_module.optimizer_g.clear_grad()
            losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                            + losses['ae'] * config['ae_loss_alpha'] \
                            + losses['wgan_g'] * config['gan_loss_alpha']
            if paddle.abs(losses['g']).item()<0.5:
                losses['g'].backward()
                trainer_module.optimizer_g.step()
              
            if paddle.abs(losses['g']).item()>1:
                item=(iteration//1000)*1000  
                print("load last model:",iteration)
                #trainer_module.resume(config['resume'],iteration=item)
                checkpoint_dir="/home/aistudio/generative-inpainting/checkpoints/imagenet/hole_benchmark/"
                trainer_module.netG.set_state_dict(paddle.load(checkpoint_dir+"gen_00"+str(item)+".pdparams"))
                last_model_name = checkpoint_dir+"dis_00"+str(iteration)+".pdparams"
                state_dict = paddle.load(last_model_name)
                trainer_module.localD.set_state_dict(state_dict['localD'])
                trainer_module.globalD.set_state_dict(state_dict['globalD'])
                

        # print(losses['d'])
        # reprod_logger = ReprodLogger()
        # reprod_logger.add("logits", (losses['d']).cpu().detach().numpy())
        # reprod_logger.save("backward_paddle.npy")

        #cv2.imwrite('%s/niter_%03d.png' % (checkpoint_path, iteration),(paddle.transpose(inpainted_result[0],[1,2,0]).numpy()+1)*256)


        #return losses

        # Log and visualization
        log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
        if iteration % config['print_iter'] == 0:
            time_count = time.time() - time_count
            speed = config['print_iter'] / time_count
            speed_msg = 'speed: %.2f batches/s ' % speed
            time_count = time.time()

            message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
            for k in log_losses:
                v = losses.get(k, 0.)
                #writer.add_scalar(k, v, iteration)
                message += '%s: %.6f ' % (k, v)
            message += speed_msg
            logger.info(message)

        if iteration % (config['viz_iter']) == 0:
            # viz_max_out = config['viz_max_out']
            # if x.shape[0] > viz_max_out:
            #     viz_images = paddle.stack([x[:viz_max_out], inpainted_result[:viz_max_out],
            #                                 offset_flow[:viz_max_out]], axis=1)
            # else:
            #     viz_images = paddle.stack([x, inpainted_result, offset_flow], axis=1)
            # viz_images = paddle.reshape(viz_images,[-1,*list(x.shape[1:])])
            # vutils.save_image(viz_images,
            #                    '%s/niter_%03d.png' % (checkpoint_path, iteration),
            #                    nrow=3 * 4,
            #                    normalize=True)

            #scipy.misc.imsave('%s/niter_%03d.png' % (checkpoint_path, iteration),viz_images.numpy()[0])
            #print(viz_images.shape)
            cv2.imwrite('%s/niter_%03d.png' % (checkpoint_path, iteration),(paddle.transpose(inpainted_result[0],[1,2,0]).numpy()+1)*256)

        # Save the model
        if iteration % (config['snapshot_save_iter']) == 0:
            trainer_module.save_model(checkpoint_path, iteration)

    #except Exception as e:  # for unexpected error logging
    #    logger.error("{}".format(e))
    #    raise e


if __name__ == '__main__':
    main()
