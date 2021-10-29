import os
import paddle
#from paddle.nn.functional import pad
import yaml
import numpy as np
from PIL import Image


import paddle.nn.functional as F

np.random.seed(123)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


def tensor_img_to_npimg(tensor_img):
    """
    Turn a tensor image with shape CxHxW to a numpy array image with shape HxWxC
    :param tensor_img:
    :return: a numpy array image with shape HxWxC
    """
    if not (paddle.is_tensor(tensor_img) and tensor_img.ndimension() == 3):
        raise NotImplementedError("Not supported tensor image. Only tensors with dimension CxHxW are supported.")
    npimg = np.transpose(tensor_img.numpy(), (1, 2, 0))
    npimg = npimg.squeeze()
    assert isinstance(npimg, np.ndarray) and (npimg.ndim in {2, 3})
    return npimg


# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    #return x.mul_(2).add_(-1)
    #mat_shape=x.shape
    #item= paddle.ones(mat_shape)
    #item=paddle.clip(item,min=2)
    #print("lxz",item)
    #x=paddle.matmul(x,paddle.to_tensor([2]))
    x=2*x
    x=x-1
    #item=paddle.clip(item,max=-1)
    #x=paddle.add(x,paddle.to_tensor([-1]))
    return x

def same_padding(images, ksizes, strides, rates):
    assert len(images.shape) == 4
    batch_size, channel, rows, cols = images.shape
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = [padding_left, padding_right, padding_top, padding_bottom]
    #images = paddle.nn.ZeroPad2d(paddings)(images)
    images = F.pad(images,paddings,mode='constant')
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.shape) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.shape

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))


    unfold = paddle.nn.Unfold(kernel_sizes=ksizes,
                             dilations=rates,
                             paddings=0,
                             strides=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def random_bbox(config, batch_size):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width, _ = config['image_shape']
    h, w = config['mask_shape']
    margin_height, margin_width = config['margin']
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if config['mask_batch_same']:
        t = np.random.randint(margin_height, maxt)
        l = np.random.randint(margin_width, maxl)
        bbox_list.append((t, l, h, w))
        bbox_list = bbox_list * batch_size
    else:
        for i in range(batch_size):
            t = np.random.randint(margin_height, maxt)
            l = np.random.randint(margin_width, maxl)
            bbox_list.append((t, l, h, w))

    return paddle.to_tensor(bbox_list, dtype="int32")


def test_random_bbox():
    image_shape = [256, 256, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    bbox = random_bbox(image_shape)
    return bbox


def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w):
    batch_size = bboxes.shape[0]
    mask = paddle.zeros((batch_size, 1, height, width), dtype=paddle.float32)
    for i in range(batch_size):
        bbox = bboxes[i]
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)
        mask[i, :, bbox[0] + delta_h:bbox[0] + bbox[2] - delta_h, bbox[1] + delta_w:bbox[1] + bbox[3] - delta_w] = 1.
    return mask


def test_bbox2mask():
    image_shape = [256, 256, 3]
    mask_shape = [128, 128]
    margin = [0, 0]
    max_delta_shape = [32, 32]
    bbox = random_bbox(image_shape)
    mask = bbox2mask(bbox, image_shape[0], image_shape[1], max_delta_shape[0], max_delta_shape[1])
    return mask


def local_patch(x, bbox_list):
    assert len(x.shape) == 4
    patches = []
    for i, bbox in enumerate(bbox_list):
        t, l, h, w = bbox
        patches.append(x[i, :, t:t + h, l:l + w])
    return paddle.stack(patches, axis=0)


def mask_image(x, bboxes, config):
    height, width, _ = config['image_shape']
    max_delta_h, max_delta_w = config['max_delta_shape']
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w)
    #if x.is_cuda:
    #    mask = mask.cuda()

    if config['mask_type'] == 'hole':
        result = x * (1. - mask)
    elif config['mask_type'] == 'mosaic':
        # TODO: Matching the mosaic patch size and the mask size
        mosaic_unit_size = config['mosaic_unit_size']
        downsampled_image = F.interpolate(x, scale_factor=1. / mosaic_unit_size, mode='nearest')
        upsampled_image = F.interpolate(downsampled_image, size=(height, width), mode='nearest')
        result = upsampled_image * mask + x * (1. - mask)
    else:
        raise NotImplementedError('Not implemented mask type.')

    return result, mask


def spatial_discounting_mask(config):
    """Generate spatial discounting mask constant.

    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        config: Config should have configuration including HEIGHT, WIDTH,
            DISCOUNTED_MASK.

    Returns:
        tf.Tensor: spatial discounting mask

    """
    gamma = config['spatial_discounting_gamma']
    height, width = config['mask_shape']
    shape = [1, 1, height, width]
    if config['discounted_mask']:
        mask_values = np.ones((height, width))
        for i in range(height):
            for j in range(width):
                mask_values[i, j] = max(
                    gamma ** min(i, height - i),
                    gamma ** min(j, width - j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 0)
    else:
        mask_values = np.ones(shape)
    spatial_discounting_mask_tensor = paddle.to_tensor(mask_values, dtype=paddle.float32)
    if config['cuda']:
        spatial_discounting_mask_tensor = spatial_discounting_mask_tensor.cuda()
    return spatial_discounting_mask_tensor


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = paddle.mean(x, axis=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = paddle.std(x, axis=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = paddle.sum(x, axis=i, keepdim=keepdim)
    return x


def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def pt_flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = paddle.to_tensor(-999)
    maxv = paddle.to_tensor(-999)
    minu = paddle.to_tensor(999)
    minv = paddle.to_tensor(999)
    maxrad = paddle.to_tensor(-1)
    if paddle.cuda.is_available(): ##########
        maxu = maxu.cuda()
        maxv = maxv.cuda()
        minu = minu.cuda()
        minv = minv.cuda()
        maxrad = maxrad.cuda()
    for i in range(flow.shape[0]):
        u = flow[i, 0, :, :]
        v = flow[i, 1, :, :]
        idxunknow = (paddle.abs(u) > 1e7) + (paddle.abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = paddle.max(maxu, paddle.max(u))
        minu = paddle.min(minu, paddle.min(u))
        maxv = paddle.max(maxv, paddle.max(v))
        minv = paddle.min(minv, paddle.min(v))
        rad = paddle.sqrt((u ** 2 + v ** 2).float()).to(paddle.int64)
        maxrad = paddle.max(maxrad, paddle.max(rad))
        u = u / (maxrad + paddle.finfo(paddle.float32).eps)
        v = v / (maxrad + paddle.finfo(paddle.float32).eps)
        # TODO: change the following to pypaddle
        img = pt_compute_color(u, v)
        out.append(img)

    return paddle.stack(out, axis=0)


def highlight_flow(flow):
    """Convert flow into middlebury color code image.
    """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def pt_highlight_flow(flow):
    """Convert flow into middlebury color code image.
        """
    out = []
    s = flow.shape
    for i in range(flow.shape[0]):
        img = np.ones((s[1], s[2], 3)) * 144.
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        for h in range(s[1]):
            for w in range(s[1]):
                ui = u[h, w]
                vi = v[h, w]
                img[ui, vi, :] = 255.
        out.append(img)
    return np.float32(np.uint8(out))


def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img


def pt_compute_color(u, v):
    h, w = u.shape
    img = paddle.zeros([3, h, w])
    if paddle.cuda.is_available():
        img = img.cuda()
    nanIdx = (paddle.isnan(u) + paddle.isnan(v)) != 0
    u[nanIdx] = 0.
    v[nanIdx] = 0.
    # colorwheel = COLORWHEEL
    colorwheel = pt_make_color_wheel()
    if paddle.cuda.is_available():
        colorwheel = colorwheel.cuda()
    ncols = colorwheel.shape[0]
    rad = paddle.sqrt((u ** 2 + v ** 2).to(paddle.float32))
    a = paddle.atan2(-v.to(paddle.float32), -u.to(paddle.float32)) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = paddle.floor(fk).to(paddle.int64)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0.to(paddle.float32)
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1]
        col1 = tmp[k1 - 1]
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1. / 255.
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = (idx != 0)
        col[notidx] *= 0.75
        img[i, :, :] = col * (1 - nanIdx).to(paddle.float32)
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def pt_make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = paddle.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 1.
    colorwheel[0:RY, 1] = paddle.arange(0, RY, dtype=paddle.float32) / RY
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 1. - (paddle.arange(0, YG, dtype=paddle.float32) / YG)
    colorwheel[col:col + YG, 1] = 1.
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 1.
    colorwheel[col:col + GC, 2] = paddle.arange(0, GC, dtype=paddle.float32) / GC
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 1. - (paddle.arange(0, CB, dtype=paddle.float32) / CB)
    colorwheel[col:col + CB, 2] = 1.
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 1.
    colorwheel[col:col + BM, 0] = paddle.arange(0, BM, dtype=paddle.float32) / BM
    col += BM
    # MR
    colorwheel[col:col + MR, 2] = 1. - (paddle.arange(0, MR, dtype=paddle.float32) / MR)
    colorwheel[col:col + MR, 0] = 1.
    return colorwheel


def is_image_file(filename):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def deprocess(img):
    img = img.add_(1).div_(2)
    return img


# get configs
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream,Loader=yaml.FullLoader)


# Get model list for resume
def get_model_list(dirname, key, iteration=0):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pdparams" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    if iteration == 0:
        last_model_name = gen_models[-1]
    else:
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    return last_model_name


if __name__ == '__main__':
    test_random_bbox()
    mask = test_bbox2mask()
    print(mask.shape)
    import matplotlib.pyplot as plt

    plt.imshow(mask, cmap='gray')
    plt.show()
