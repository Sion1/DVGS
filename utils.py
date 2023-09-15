import errno
import os
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from PIL import Image
# from pytorch-gradcam import ActivationsAndGradients
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pandas as pd


class DataLoader(Dataset):
    def __init__(self, root, image_files, labels, transform=None):
        self.root = root
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        # read the iterable image
        img_pil = Image.open(os.path.join(self.root, self.image_files[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        # label
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def str2bool(str):
    return True if str.lower() == 'true' else False


def remove_zero(attr_prototype, attr_weight):
    #### broadcast attr weight
    attr_weight_broad = attr_weight.reshape(attr_weight.shape[0], 1, attr_weight.shape[1]).repeat(1, attr_prototype.shape[0], 1)
    #### broadcast attr
    attr = attr_prototype.unsqueeze(dim=0).repeat(attr_weight.shape[0], 1, 1)
    #### non zero index
    index_non_zero = attr.nonzero()
    index_non_zero_0 = index_non_zero[:, 0]
    index_non_zero_1 = index_non_zero[:, 1]
    index_non_zero_2 = index_non_zero[:, 2]
    index_non_zero = (index_non_zero_0, index_non_zero_1, index_non_zero_2)
    #### zero index
    index_zero = (attr == 0).nonzero()
    index_zero_0 = index_zero[:, 0]
    index_zero_1 = index_zero[:, 1]
    index_zero_2 = index_zero[:, 2]
    index_zero = (index_zero_0, index_zero_1, index_zero_2)
    value = torch.tensor(0.0).to('cuda:0')
    new_attr_weight = attr_weight_broad.index_put(index_zero, value)
    for i in range(attr_prototype.shape[0]):
        index_norm_non_zero = attr_prototype[i].nonzero().squeeze()
        norm_attr = F.softmax(new_attr_weight[:, i, index_norm_non_zero], dim=-1)
        new_attr_weight[:, i, index_norm_non_zero] = norm_attr
    return new_attr_weight


def get_w2v(dataset):
    model_name = 'word2vec-google-news-300'  # best model
    model = KeyedVectors.load_word2vec_format(
        datapath(f'/media/user/F338-5CBE/XL/GoogleNews-vectors-negative300.bin.gz'),
        binary=True)
    dim_w2v = 300
    print('Done loading model')
    replace_word = [('Auklet', 'auklet'), ('Frigatebird', 'Frigate bird'), ('Glaucous', 'glaucous'), ('Slaty', 'slaty'),
                    ('Violetear', 'violet ear'), ('Pomarine', 'pomarine'), ('Ovenbird', 'ovenbird'), ('Sayornis', 'sayornis'),
                    ('Geococcyx', 'geococcyx'), ('and', ' '), ('Waterthrush', 'water thrush'), ('cockaded', 'cockade'), ('Yellowthroat', 'yellow throat')]
    path = '/media/user/F338-5CBE/Dataset/CUB_200_2011/CUB_200_2011/classes.txt'
    df = pd.read_csv(path, sep=' ', header=None, names=['idx', 'des'])
    des = df['des'].values
    new_des = [' '.join(i.split('_')) for i in des]
    new_des = [' '.join(i.split('-')) for i in new_des]
    new_des = [' '.join(i.split('::')) for i in new_des]
    new_des = [i.split('(')[0] for i in new_des]
    new_des = [i[4:] for i in new_des]
    for pair in replace_word:
        for idx, s in enumerate(new_des):
            new_des[idx] = s.replace(pair[0], pair[1])
    all_w2v = []
    for s in new_des:
        words = s.split(' ')
        if words[-1] == '':  # remove empty element
            words = words[:-1]
        w2v = np.zeros(dim_w2v)
        for w in words:
            try:
                w2v += model[w]
            except Exception as e:
                print(e)
        all_w2v.append(w2v[np.newaxis, :])
    all_w2v = np.concatenate(all_w2v, axis=0)
    return all_w2v


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img


if __name__ == '__main__':
    print(get_w2v('CUB'))