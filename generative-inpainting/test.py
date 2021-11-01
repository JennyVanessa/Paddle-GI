import os
import random
from tqdm import tqdm
import shutil
from argparse import ArgumentParser

import paddle
import paddle.nn as nn
import paddle.vision
import numpy as np

from tester import Trainer
from datatest.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image
from utils.logger import get_logger
import cv2
 



parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/test.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')

class TVLoss(nn.Layer):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.shape[0]
        h_x = x.shape[2]
        w_x = x.shape[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = paddle.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = paddle.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.shape[1]*t.shape[2]*t.shape[3]


def main():
    args = parser.parse_args()
    config = get_config(args.config)
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids

    checkpoint_path = os.path.join('checkpoints',
                                   config['dataset_name'],
                                   config['mask_type'] + '_' + config['expname'])
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    shutil.copy(args.config, os.path.join(checkpoint_path, os.path.basename(args.config)))
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
    train_loader = paddle.io.DataLoader(dataset=train_dataset,
                                                batch_size=config['batch_size'],
                                                shuffle=False,
                                                num_workers=config['num_workers'],
                                                use_shared_memory=True,
                                                use_buffer_reader=True)

    # Define the trainer
    trainer = Trainer(config)
    logger.info("\n{}".format(trainer.netG))
    logger.info("\n{}".format(trainer.localD))
    logger.info("\n{}".format(trainer.globalD))

    trainer_module=trainer

    # Get the resume iteration to restart training
    start_iteration = trainer_module.resume(config['resume']) if config['resume'] else 1

    iterable_train_loader = iter(train_loader)

    l1list=[]
    l2list=[]
    PSNRlist=[]
    #tvlist=[]

    for iteration in tqdm(range(10000)):    
        try:
            ground_truth = next(iterable_train_loader)
            ground_truth = paddle.to_tensor(ground_truth)
        except StopIteration:
            iterable_train_loader = iter(train_loader)
            ground_truth = next(iterable_train_loader)
            ground_truth = paddle.to_tensor(ground_truth)

        bboxes = random_bbox(config, batch_size=ground_truth.shape[0])
        x, mask = mask_image(ground_truth, bboxes, config)

        inpainted_result= trainer(x, bboxes, mask, ground_truth)

        inpainted_result1=((inpainted_result+1)/2)*255
        ground_truth1=((ground_truth+1)/2)*255

        l1loss=nn.L1Loss()
        l2loss=nn.MSELoss()
        #tvloss=TVLoss()
        PSNR=10*paddle.log10(255*255/l2loss(inpainted_result1,ground_truth1))
        l1list.append(l1loss(inpainted_result,ground_truth).item())
        l2list.append(l2loss(inpainted_result,ground_truth).item())
        PSNRlist.append(PSNR.item())
        #tvlist.append(tvloss(inpainted_result).item())

        img=((paddle.transpose(inpainted_result[0],[1,2,0]).numpy()+1)/2)*255
        save_path="/home/aistudio/testoutput/"+str(iteration)+".png"

        img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        cv2.imwrite(save_path,img1)
        
    print("l1:",np.mean(np.array(l1list)))
    print("l2:",np.mean(np.array(l2list)))
    print("PSNR:",np.mean(np.array(PSNRlist)))
    #print("TVLoss:",np.mean(np.array(tvlist)))
    

if __name__ == '__main__':
    main()
