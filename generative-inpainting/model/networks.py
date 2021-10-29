
#from torch.nn.modules import padding
#from torchvision import transforms
#from torchvision import utils as vutils

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.utils import spectral_norm as spectral_norm_fn
from paddle.nn.utils import weight_norm as weight_norm_fn
from PIL import Image

from utils.tools import extract_image_patches, flow_to_image, \
    reduce_mean, reduce_sum, default_loader, same_padding


class Generator(nn.Layer):
    def __init__(self, config, use_cuda, device_ids):
        super(Generator, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ngf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.coarse_generator = CoarseGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)
        self.fine_generator = FineGenerator(self.input_dim, self.cnum, self.use_cuda, self.device_ids)

    def forward(self, x, mask):
        x_stage1 = self.coarse_generator(x, mask)
        x_stage2, offset_flow = self.fine_generator(x, x_stage1, mask)
        return x_stage1, x_stage2, offset_flow


class CoarseGenerator(nn.Layer):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum*2, 3, 2, 1)
        self.conv3 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        self.conv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        self.conv11 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.conv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.conv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.conv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.conv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.conv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, x, mask):
        # For indicating the boundaries of images
        ones = paddle.ones([x.shape[0], 1, x.shape[2], x.shape[3]],dtype="float32") ###size
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # 5 x 256 x 256
        #x = self.conv1(paddle.cat([x, ones, mask], dim=1))####
        x = self.conv1(paddle.concat([x, ones, mask], axis=1))
        x = self.conv2_downsample(x)
        # cnum*2 x 128 x 128
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        # cnum*4 x 64 x 64
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum*2 x 128 x 128
        x = self.conv13(x)
        x = self.conv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        # cnum x 256 x 256
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        # 3 x 256 x 256
        x_stage1 = paddle.clip(x, -1., 1.)
        #x_stage1 = F.sigmoid(x)

        return x_stage1


class FineGenerator(nn.Layer):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(FineGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        # 3 x 256 x 256
        self.conv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.conv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.conv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.conv4_downsample = gen_conv(cnum*2, cnum*2, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.conv5 = gen_conv(cnum*2, cnum*4, 3, 1, 1)
        self.conv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1)

        self.conv7_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 2, rate=2)
        self.conv8_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 4, rate=4)
        self.conv9_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 8, rate=8)
        self.conv10_atrous = gen_conv(cnum*4, cnum*4, 3, 1, 16, rate=16)

        # attention branch
        # 3 x 256 x 256
        self.pmconv1 = gen_conv(input_dim + 2, cnum, 5, 1, 2)
        self.pmconv2_downsample = gen_conv(cnum, cnum, 3, 2, 1)
        # cnum*2 x 128 x 128
        self.pmconv3 = gen_conv(cnum, cnum*2, 3, 1, 1)
        self.pmconv4_downsample = gen_conv(cnum*2, cnum*4, 3, 2, 1)
        # cnum*4 x 64 x 64
        self.pmconv5 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv6 = gen_conv(cnum*4, cnum*4, 3, 1, 1, activation='relu')
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=self.use_cuda, device_ids=self.device_ids)
        self.pmconv9 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.pmconv10 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv11 = gen_conv(cnum*8, cnum*4, 3, 1, 1)
        self.allconv12 = gen_conv(cnum*4, cnum*4, 3, 1, 1)
        self.allconv13 = gen_conv(cnum*4, cnum*2, 3, 1, 1)
        self.allconv14 = gen_conv(cnum*2, cnum*2, 3, 1, 1)
        self.allconv15 = gen_conv(cnum*2, cnum, 3, 1, 1)
        self.allconv16 = gen_conv(cnum, cnum//2, 3, 1, 1)
        self.allconv17 = gen_conv(cnum//2, input_dim, 3, 1, 1, activation='none')

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        #ones = paddle.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        ones = paddle.ones([xin.shape[0], 1, xin.shape[2], xin.shape[3]],dtype="float32")
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        # conv branch
        xnow = paddle.concat([x1_inpaint, ones, mask], axis=1)
        x = self.conv1(xnow)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x_hallu = x
        # attention branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        x = self.pmconv6(x)
        x, offset_flow = self.contextul_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = paddle.concat([x_hallu, pm], axis=1)
        # merge two branches
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv13(x)
        x = self.allconv14(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.allconv15(x)
        x = self.allconv16(x)
        x = self.allconv17(x)
        x_stage2 = paddle.clip(x, -1., 1.)
        #x_stage2 = F.sigmoid(x)

        return x_stage2, offset_flow


class ContextualAttention(nn.Layer):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=False, use_cuda=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.shape)   # b*c*h*w
        raw_int_bs = list(b.shape)   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate*self.stride,
                                               self.rate*self.stride],
                                      rates=[1, 1],
                                      padding='same') # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        #raw_w = raw_w.reshape(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = paddle.reshape(raw_w,[raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1])
        #raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w = paddle.transpose(raw_w,[0,4,1,2,3])
        #raw_w_groups = paddle.split(raw_w, 1, axis=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1./self.rate, mode='nearest')
        int_fs = list(f.shape)     # b*c*h*w
        int_bs = list(b.shape)
        #f_groups = paddle.split(f, 1, axis=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        #w = w.reshape(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = paddle.reshape(w,[int_bs[0], int_bs[1], self.ksize, self.ksize, -1])
        #w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w = paddle.transpose(w,[0,4,1,2,3])
        #w_groups = paddle.split(w, 1, axis=0)


        # process mask
        if mask is None:
            mask = paddle.zeros([int_bs[0], 1, int_bs[2], int_bs[3]])
            if self.use_cuda:
                mask = mask.cuda()
        else:
            mask = F.interpolate(mask, scale_factor=1./(4*self.rate), mode='nearest')
        int_ms = list(mask.shape)
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # m shape: [N, C, k, k, L]
        #m = m.reshape(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = paddle.reshape(m,[int_ms[0], int_ms[1], self.ksize, self.ksize, -1])
        #m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = paddle.transpose(m,[0,4,1,2,3])
        m = m[0]    # m shape: [L, C, k, k]

        # mm shape: [L, 1, 1, 1]
        #mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.).to(paddle.float32)
        mm = reduce_mean(m, axis=[1, 2, 3], keepdim=True)==0.
        mm = mm.astype("float32")

        #mm = mm.permute(1, 0, 2, 3) # mm shape: [1, L, 1, 1]
        mm = paddle.transpose(mm,[1,0,2,3])

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range
        fuse_weight = paddle.eye(k)#.reshape(1, 1, k, k)  # 1*1*k*k
        fuse_weight = paddle.reshape(fuse_weight,[1,1,k,k])
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for lxz in range(f.shape[0]):
            xi=f[0].unsqueeze(0)
            wi=w[0].unsqueeze(0)
            raw_wi=raw_w[0].unsqueeze(0)
        #for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = paddle.to_tensor([1e-4],dtype="float32")
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = paddle.sqrt(reduce_sum(paddle.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)   # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                #yi = yi.reshape(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = paddle.reshape(yi,[1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3]])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                #yi = yi.contiguous().reshape(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                #yi = yi.reshape(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = paddle.reshape(yi,[1,int_bs[2], int_bs[3], int_fs[2], int_fs[3]])
                #yi = yi.permute(0, 2, 1, 4, 3)
                yi = paddle.transpose(yi,[0,2,1,4,3])
                #yi = yi.contiguous().reshape(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                #yi = yi.reshape(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = paddle.reshape(yi,[1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3]])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                #yi = yi.contiguous().reshape(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                #yi = yi.reshape(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = paddle.reshape(yi,[1, int_bs[3], int_bs[2], int_fs[3], int_fs[2]])
                #yi = yi.permute(0, 2, 1, 4, 3).contiguous()
                yi = paddle.transpose(yi,[0,2,1,4,3])
            #yi = yi.reshape(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            yi = paddle.reshape(yi,[1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3]])
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, axis=1)
            yi = yi * mm  # [1, L, H, W]

            offset = paddle.argmax(yi, axis=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                #offset = ((offset + 1).float() * times - 1).to(paddle.int64)
                offset = (offset + 1).float() * times - 1
                offset = offset.astype("int64")

            offset = paddle.concat([offset//int_fs[3], offset%int_fs[3]], axis=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv2d_transpose(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64) ################################
            y.append(yi)
            offsets.append(offset)

        y = paddle.concat(y, axis=0)  # back to the mini-batch
        #y.contiguous().view(raw_int_fs)
        y.reshape(raw_int_fs)

        offsets = paddle.concat(offsets, axis=0)
        #offsets = offsets.reshape(int_fs[0], 2, *int_fs[2:])
        offsets = paddle.reshape(offsets,[int_fs[0], 2, *int_fs[2:]])

        # case1: visualize optical flow: minus current position

        h_add = paddle.reshape(paddle.arange(int_fs[2]),[1, 1, int_fs[2], 1])
        h_add= paddle.expand(h_add,[int_fs[0], -1, -1, int_fs[3]])
        #h_add = paddle.arange(int_fs[2]).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = paddle.reshape(paddle.arange(int_fs[3]),[1, 1, 1, int_fs[3]])
        w_add = paddle.expand(w_add,[int_fs[0], -1, int_fs[2], -1])
        #w_add = paddle.arange(int_fs[3]).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        ref_coordinate = paddle.concat([h_add, w_add], axis=1)
        if self.use_cuda:
            ref_coordinate = ref_coordinate.cuda()

        offsets = offsets - ref_coordinate
        # flow = pt_flow_to_image(offsets)

        #flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        #flow = paddle.to_tensor(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = paddle.to_tensor(flow_to_image(offsets.transpose([0, 2, 3, 1]).cpu().numpy())) / 255.
        #flow = flow.permute(0, 3, 1, 2)
        flow= paddle.transpose(flow,[0,3,1,2])
        if self.use_cuda:
            flow = flow.cuda()
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate*4, mode='nearest')

        return y, flow


class LocalDis(nn.Layer):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(LocalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*8*8, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        #x = x.reshape(x.size()[0], -1)
        x = paddle.reshape(x,[x.shape[0],-1])
        x = self.linear(x)

        return x


class GlobalDis(nn.Layer):
    def __init__(self, config, use_cuda=True, device_ids=None):
        super(GlobalDis, self).__init__()
        self.input_dim = config['input_dim']
        self.cnum = config['ndf']
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):
        x = self.dis_conv_module(x)
        #x = x.reshape(x.shape[0], -1)
        x = paddle.reshape(x,[x.shape[0], -1])
        x = self.linear(x)

        return x


class DisConvModule(nn.Layer):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1,
             activation='elu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Layer):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        self.pad_type = pad_type
        self.padding = padding
        # initialize padding
        #if pad_type == 'reflect':
            #self.pad = nn.ReflectionPad2d(padding)
        #    self.pad = F.pad(mode="")
        #elif pad_type == 'replicate':
        #    self.pad = nn.ReplicationPad2d(padding)
        #elif pad_type == 'zero':
        #    self.pad = nn.ZeroPad2d(padding)
        #elif pad_type == 'none':
        #    self.pad = None
        #else:
        #    assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2D(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2D(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.Conv2DTranspose(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation)
                                           #bias=self.use_bias)###############################
        else:
            self.conv = nn.Conv2D(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation)
                                  #bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        #if self.pad:
        if self.padding !=0:
            if self.pad_type == 'reflect':
                x=F.pad(x,self.padding,mode='reflect')
                x = self.conv(x)
            elif self.pad_type == 'replicate':
                x=F.pad(x,self.padding,mode='replicate')
                x = self.conv(x)
            elif self.pad_type == 'zero':
                x=F.pad(x,self.padding,mode='constant')
                x = self.conv(x)


        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



if __name__ == "__main__":
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--imageA', default='', type=str, help='Image A as background patches to reconstruct image B.')
    #parser.add_argument('--imageB', default='', type=str, help='Image B is reconstructed with image A.')
    #parser.add_argument('--imageOut', default='result.png', type=str, help='Image B is reconstructed with image A.')
    #args = parser.parse_args()
    #test_contextual_attention(args)


    lxz=ContextualAttention()
    print(lxz)
