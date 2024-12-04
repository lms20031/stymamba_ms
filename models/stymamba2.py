import torch
import torch.nn.functional as F
from torch import nn
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from function import normal
from function import calc_mean_std
from models.ViT_helper import to_2tuple
import clip  # Make sure CLIP is installed
from torchvision.transforms import Normalize

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding with optional patch shuffling
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, shuffle_patches=False):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)

        if shuffle_patches:
            x = self.shuffle_patches(x)

        return x

    def shuffle_patches(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)  # Flatten the spatial dimensions
        idx = torch.randperm(H * W)  # Generate a random permutation of patch indices
        x = x[:, :, idx]  # Shuffle patches
        x = x.view(B, C, H, W)  # Reshape back to the original dimensions
        return x

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self, encoder, decoder, PatchEmbed: PatchEmbed, mambanet, args, name_info, device):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.mambanet = mambanet
        hidden_dim = mambanet.d_model       
        self.decode = decoder
        self.embedding = PatchEmbed
        self.name_info = name_info

        # Initialize CLIP model
        self.clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # CLIP normalization parameters
        self.clip_normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                        std=[0.26862954, 0.26130258, 0.27577711])

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor, style_labels):
        content_input = samples_c
        style_input = samples_s
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        style = self.embedding(samples_s.tensors, shuffle_patches=True)
        content = self.embedding(samples_c.tensors)

        pos_s = None
        pos_c = None

        mask = None
        hs = self.mambanet(style, mask, content, pos_c, pos_s)
        Ics = self.decode(hs)

        # ----- CLIP Loss Computation -----
        #normalization for encoding
        Ics_clamped = Ics.clamp(0, 1)
        Ics_normalized = self.clip_normalize(Ics_clamped)
        Ics_resized = F.interpolate(Ics_normalized, size=(224, 224), mode='bilinear', align_corners=False)

        # Encode image using CLIP
        with torch.no_grad():
            image_features = self.clip_model.encode_image(Ics_resized)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Retrieve the corresponding text for each style label
        style_labels_list = style_labels.tolist()  
        texts = [self.name_info[label] for label in style_labels_list]
        text_tokens = clip.tokenize(texts).to(Ics.device)  
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        cosine_sim = torch.sum(image_features * text_features, dim=-1) 
        clip_loss = 1 - cosine_sim.mean()
 

        # ----- Contrastive Loss Computation -----
        # Encode style images using CLIP image encoder
        with torch.no_grad():
            style_images = samples_s.tensors.clamp(0, 1)
            style_images_normalized = self.clip_normalize(style_images)
            style_images_resized = F.interpolate(style_images_normalized, size=(224, 224), mode='bilinear', align_corners=False)
            clip_encoded_s = self.clip_model.encode_image(style_images_resized)
            clip_encoded_s = clip_encoded_s / clip_encoded_s.norm(dim=-1, keepdim=True)

        similarity_matrix = clip_encoded_s @ image_features.T  # Shape: (batch_size, batch_size)
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size).to(Ics.device)

        # Compute cross-entropy loss for image-to-style and style-to-image
        contrastive_loss_i2s = F.cross_entropy(similarity_matrix, labels)
        contrastive_loss_s2i = F.cross_entropy(similarity_matrix.T, labels)
        contrastive_loss = (contrastive_loss_i2s + contrastive_loss_s2i) / 2


        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1])) + \
                self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))

        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        Icc = self.decode(self.mambanet(content, mask, content, pos_c, pos_c))
        Iss = self.decode(self.mambanet(style, mask, style, pos_s, pos_s))

        loss_lambda1 = self.calc_content_loss(Icc, content_input) + self.calc_content_loss(Iss, style_input)

        Icc_feats = self.encode_with_intermediate(Icc)
        Iss_feats = self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0]) + self.calc_content_loss(Iss_feats[0], style_feats[0])

        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i]) + self.calc_content_loss(Iss_feats[i], style_feats[i])

        return Ics, loss_c, loss_s, loss_lambda1, loss_lambda2, clip_loss, contrastive_loss