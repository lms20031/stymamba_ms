import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import json


import models.mambanet2 as mambanet
import models.stymamba2 as StyMamba

from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image

def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root, self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root, file_name)):
                    self.paths.append(self.root + "/" + file_name + "/" + file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'

class StyleDataset(data.Dataset):
    def __init__(self, root_dir, text_path, path_label_path, transform=None):
        """
        Args:
            root_dir (str): 이미지 파일들이 위치한 기본 디렉토리 경로.
            text_path (str): 라벨 ID와 클래스 이름이 매핑된 텍스트 파일 경로.
            path_label_path (str): 이미지 경로와 라벨 ID가 매핑된 JSON 파일 경로.
            transform (callable, optional): 이미지에 적용할 변환 함수.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.label_map = self._load_label_map(text_path)
        self.image_label_map = self._load_image_label_map(path_label_path)
        self.image_paths = list(self.image_label_map.keys())
        self.labels = list(self.image_label_map.values())

    def _load_label_map(self, text_path):
        label_map = {}
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        label_id, class_name = parts[0], ' '.join(parts[1:])
                        label_map[label_id] = class_name
        except Exception as e:
            print(f"Error while loading label map: {e}")
        return label_map

    def _load_image_label_map(self, path_label_path):
        try:
            with open(path_label_path, 'r', encoding='utf-8') as file:
                image_label_map = json.load(file)
            return image_label_map
        except Exception as e:
            print(f"Error while loading image label: {e}")
            return {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("index out of range.")

        image_rel_path = os.path.join(self.root_dir, 'wikiart', self.image_paths[idx] )
        label_id = self.labels[idx]   
        class_name = self.label_map.get(label_id, "Unknown")

        if not os.path.isabs(image_rel_path):
            image_path = os.path.join(self.root_dir, image_rel_path)
        else:
            image_path = image_rel_path

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error while loading image: {e} (path: {image_path})")

        if self.transform:
            image = self.transform(image)

        return int(label_id), image

def adjust_learning_rate(optimizer, iteration_count, args):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count, args):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', default='/hdd/dataset/coco_nano/train', type=str,   
                        help='Directory path to a batch of content images')
    # wikiart dataset crawled from https://www.wikiart.org/
    parser.add_argument('--style_dir', default='../monet_jpg', type=str,  
                        help='Directory path to a batch of style images')
    # download the pretrained vgg checkpoint
    parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  

    # training options
    parser.add_argument('--save_dir', default='./experiments',
                        help='Directory to save the model')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=float, default=1e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--style_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=7.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--continue_train', action='store_true',
                        help="Continue training from a saved model checkpoint")
    parser.add_argument('--checkpoint', type=str, default='./experiments/model_iter_10000.pth',
                        help="Path to the saved model checkpoint")

    args = parser.parse_args()

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    vgg = StyMamba.vgg
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:44])

    decoder = StyMamba.decoder
    embedding = StyMamba.PatchEmbed()

    mamba = mambanet.MambaNet()

    with open(os.path.join(args.style_dir,'artist_mapping.json'), 'r', encoding='utf-8') as f:
        name = json.load(f) #key: artist name mapped file
    name_info = {int(k): v for k, v in name.items()}

    with torch.no_grad():
        network = StyMamba.StyTrans(vgg, decoder, embedding, mamba, args, name_info, device)
    network.train()

    if args.continue_train and args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        network.module.mambanet.load_state_dict(checkpoint['mambanet'])
        network.module.decode.load_state_dict(checkpoint['decoder'])
        network.module.embedding.load_state_dict(checkpoint['embedding'])

    network.to(device)
    network = nn.DataParallel(network, device_ids=[0])
    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(args.content_dir, content_tf)
    text_path = os.path.join(args.style_dir,'artist_class.txt')
    path_label = os.path.join(args.style_dir,'path_label.json') 
    style_dataset = StyleDataset(args.style_dir, text_path, path_label, style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=args.n_threads)) 

    optimizer = torch.optim.Adam([
        {'params': network.module.mambanet.parameters()},
        {'params': network.module.decode.parameters()},
        {'params': network.module.embedding.parameters()},        
    ], lr=args.lr)

    if not os.path.exists(args.save_dir + "/test"):
        os.makedirs(args.save_dir + "/test")

    for i in tqdm(range(args.max_iter)):
        if i < 1e4:
            warmup_learning_rate(optimizer, iteration_count=i, args=args)
        else:
            adjust_learning_rate(optimizer, iteration_count=i, args=args)

        content_images = next(content_iter).to(device)
        style_labels, style_images = next(style_iter)
        style_labels = style_labels.to(device)
        style_images = style_images.to(device)
        out, loss_c, loss_s, l_identity1, l_identity2, clip_loss, contrastive_loss = network(content_images, style_images, style_labels)

        loss_c = args.content_weight * loss_c
        loss_s = args.style_weight * loss_s
        loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1) + clip_loss + contrastive_loss

        if i % 250 == 0:
            tqdm.write(f'Epoch: {i}, Total_loss: {loss.sum().item():.4f}, Content_loss: {loss_c.sum().item():.4f}, Style_loss: {loss_s.sum().item():.4f}, Identity_L1: {l_identity1.sum().item():.4f}, Identity_L2: {l_identity2.sum().item():.4f},  Clip_loss: {clip_loss.sum().item():.4f},  Contrastive_loss: {contrastive_loss.sum().item():.4f}')
        
        if i % 1000 == 0:
            output_name = '{:s}/test/{:s}{:s}'.format(
                            args.save_dir, str(i), ".jpg"
                        )
            out = torch.cat((content_images, out), 0)
            out = torch.cat((style_images, out), 0)
            save_image(out, output_name)

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            # Prepare the state dictionaries
            mambanet_state_dict = network.module.mambanet.state_dict()
            decoder_state_dict = network.module.decode.state_dict()
            embedding_state_dict = network.module.embedding.state_dict()

            # Move all tensors to CPU
            for key in mambanet_state_dict.keys():
                mambanet_state_dict[key] = mambanet_state_dict[key].to(torch.device('cpu'))
            for key in decoder_state_dict.keys():
                decoder_state_dict[key] = decoder_state_dict[key].to(torch.device('cpu'))
            for key in embedding_state_dict.keys():
                embedding_state_dict[key] = embedding_state_dict[key].to(torch.device('cpu'))

            # Combine into a single dictionary
            combined_state_dict = {
                'mambanet': mambanet_state_dict,
                'decoder': decoder_state_dict,
                'embedding': embedding_state_dict
            }

            # Save the combined dictionary
            torch.save(combined_state_dict, '{:s}/model_iter_{:d}.pth'.format(args.save_dir, i + 1))

if __name__ == '__main__':
    main()
