import os
import random
from argparse import ArgumentParser

# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
# import torchvision.utils as vutils

import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
import cv2

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--image', type=str)
parser.add_argument('--mask', type=str, default='')
parser.add_argument('--output', type=str, default='output.png')
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--checkpoint_path', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

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

def variation_loss(image, ksize=1):
    dh = image[:, :, :-ksize, :] - image[:, :, ksize:, :]
    dw = image[:, :, :, :-ksize] - image[:, :, :, ksize:]
    return (paddle.mean(paddle.abs(dh)) + paddle.mean(paddle.abs(dw)))

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    # if cuda:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
    #     device_ids = list(range(len(device_ids)))
    #     config['gpu_ids'] = device_ids
    #     cudnn.benchmark = True

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    paddle.seed(args.seed)
    # if cuda:
    #     torch.cuda.manual_seed_all(args.seed)

    print("Configuration: {}".format(config))

    try:  # for unexpected error logging
        #with torch.no_grad():   # enter no grad context
        if is_image_file(args.image):
            if args.mask and is_image_file(args.mask):
                # Test a single masked image with a given mask
                x = default_loader(args.image)
                mask = default_loader(args.mask)
                x = transforms.Resize(config['image_shape'][:-1])(x)
                x = transforms.CenterCrop(config['image_shape'][:-1])(x)
                mask = transforms.Resize(config['image_shape'][:-1])(mask)
                mask = transforms.CenterCrop(config['image_shape'][:-1])(mask)
                x = transforms.ToTensor()(x)
                mask = transforms.ToTensor()(mask)[0].unsqueeze(0)
                x = normalize(x)
                x = x * (1. - mask)
                x = x.unsqueeze(0)
                mask = mask.unsqueeze(0)
            elif args.mask:
                raise TypeError("{} is not an image file.".format(args.mask))
            else:
                # Test a single ground-truth image with a random mask
                ground_truth = default_loader(args.image)
                ground_truth = transforms.Resize(config['image_shape'][:-1])(ground_truth)
                ground_truth = transforms.CenterCrop(config['image_shape'][:-1])(ground_truth)
                ground_truth = transforms.ToTensor()(ground_truth)
                ground_truth = normalize(ground_truth)
                ground_truth = ground_truth.unsqueeze(dim=0)
                bboxes = random_bbox(config, batch_size=ground_truth.size(0))
                x, mask = mask_image(ground_truth, bboxes, config)

            # Set checkpoint path
            if not args.checkpoint_path:
                checkpoint_path = os.path.join('checkpoints',
                                                config['dataset_name'],
                                                config['mask_type'] + '_' + config['expname'])
            else:
                checkpoint_path = args.checkpoint_path

            # Define the trainer
            netG = Generator(config['netG'], cuda, device_ids)
            # Resume weight
            #last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
            mod=paddle.load("/home/aistudio/generative-inpainting/checkpoints/imagenet/hole_benchmark/gen_00107000.pdparams")
            netG.set_state_dict(mod)
            model_iteration = int(107000)
            print("Resume from {} at iteration {}".format(checkpoint_path, model_iteration))

            # if cuda:
            #     netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
            #     x = x.cuda()
            #     mask = mask.cuda()

            # Inference
            x1, x2, offset_flow = netG(x, mask)
            inpainted_result = x2 * mask + x * (1. - mask)

            l1loss=nn.L1Loss()
            l2loss=nn.MSELoss()
            #tvloss=TVLoss()
            PSNR=10*paddle.log10(1/l2loss(inpainted_result,x))
            
            print(l1loss(inpainted_result,x))
            print(l2loss(inpainted_result,x))
            print(PSNR)
            print(variation_loss(inpainted_result))

            #vutils.save_image(inpainted_result, args.output, padding=0, normalize=True)
            cv2.imwrite(args.output,(paddle.transpose(inpainted_result[0],[1,2,0]).numpy()+1)*128)
            #cv2.imwrite(args.output,(inpainted_result[0][0].numpy()+1)*256)
            print("Saved the inpainted result to {}".format(args.output))
            # if args.flow:
            #     vutils.save_image(offset_flow, args.flow, padding=0, normalize=True)
            #     print("Saved offset flow to {}".format(args.flow))
        else:
            raise TypeError("{} is not an image file.".format)
        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    main()
