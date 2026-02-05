from math import sqrt
import torch
import math
import torch.nn as nn
import matplotlib.pyplot as plt
from accelerate.test_utils.scripts.test_script import print_on
from llava.mm_utils import divide_to_patches

from llava.mm_utils import resize_and_pad_image
from PIL import Image
import matplotlib.pyplot as plt
import re
from transformers import CLIPImageProcessor, CLIPVisionModel


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # query: low resolution
        # key_value: high resolution
        attn_output, _ = self.attn(query, key_value, key_value)
        return self.norm(query + attn_output)  # 残差连接


# image = Image.open('../../../images/llava_logo.png').convert('RGB')
# image = image.resize((256, 256))
#
# new_image = resize_and_pad_image(image, (512, 512))
# plt.figure()
#
# plt.subplot(1, 2, 1)
# plt.title("Original 256x256")
# plt.imshow(image)
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title("Padded to 512x512")
# plt.imshow(new_image)
# plt.axis('off')
#
# plt.show()
#
# image.show()
# new_image.show()

# projector_type = 'mlp2x_gelu'
# mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
# print(mlp_gelu_match)
# mlp_depth = int(mlp_gelu_match.group(1))
# print(mlp_depth)

# image = Image.open('/Users/zhangjingrui/PycharmProjects/LLaVA/llava/model/multimodal_projector/a.JPG').convert('RGB')
# image.show()
# image_divided = divide_to_patches(image, patch_size=512)
# for image_patch in image_divided:
#     image_patch.show()
#
image_processor = CLIPImageProcessor.from_pretrained('../../../lmsys/clip-vit-large-patch14-336')
# print(image_processor)


# w,h = 336, 336
# image = image.resize((w,h))
# image.show()
# image_high_res = image.resize((w*2,h*2))
# patches = divide_to_patches(image_high_res, image_processor.crop_size['height'])
#
# for patch in patches:
#     patch.show()

# print(type(patches[0]))
# image_patches = [image] + patches
# image_patches = [image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in image_patches]
# image = torch.stack(image_patches, dim=0)
# print(image.shape)

random_tensor = torch.rand((2,5,3,336,336))
# concat_images = torch.cat([image for image in random_tensor], dim=0)
# print(concat_images.shape)



# tensor = image_processor(images=image, return_tensors='pt')
# print(tensor['pixel_values'].shape)
device_map = torch.device('mps')
vision_tower = CLIPVisionModel.from_pretrained('../../../lmsys/clip-vit-large-patch14-336', device_map=device_map)
random_tensor = random_tensor.view(-1, 3, 336, 336)
random_tensor = random_tensor.to(device_map)
output = vision_tower(random_tensor, output_hidden_states=True)
u_feature = output.hidden_states
# print(u_feature)
u_feature = list(u_feature)

for listx in u_feature:
    print(listx.shape)

# print(u_feature.shape)
# output = vision_tower(tensor['pixel_values'].to(device_map), output_attentions=True)
# print(output.attentions)
attn = output.attentions
# for atn in attn:
#     print(atn.shape)
#
#

# def visualize_cls_attention_per_layer(attentions, patch_grid_size=(24, 24), head_index=0):
#     """
#     可视化每一层中 [CLS] token 对所有 patch 的注意力热图
#
#     Args:
#         attentions (list[Tensor]): 每一层的 attention 输出，形状为 [1, num_heads, 577, 577]
#         patch_grid_size (tuple): patch reshape 成图像时的网格大小，默认为 24x24
#         head_index (int): 选择第几个注意力头来可视化
#     """
#     num_layers = len(attentions)
#     fig_cols = 6
#     fig_rows = math.ceil(num_layers / fig_cols)
#
#     plt.figure(figsize=(fig_cols * 3, fig_rows * 3))
#
#     for i, attn in enumerate(attentions):
#         # 取出第 i 层、第 head_index 个注意力头，[CLS] token 是第 0 个
#         cls_attn = attn[0, head_index, 0, 1:]  # shape: (576,)
#
#         # reshape 成 24x24 热图
#         cls_attn_map = cls_attn.reshape(patch_grid_size).detach().cpu().numpy()
#
#         # 画图
#         plt.subplot(fig_rows, fig_cols, i + 1)
#         plt.imshow(cls_attn_map, cmap='viridis')
#         plt.title(f"Layer {i+1}")
#         plt.axis('off')
#
#     plt.suptitle(f"[CLS] Attention Maps from Head {head_index}", fontsize=16)
#     plt.tight_layout()
#     plt.show()
#
# # visualize_cls_attention_per_layer(attn, patch_grid_size=(24, 24), head_index=3)
#
# for head in range(3, 10):
#     visualize_cls_attention_per_layer(attn, patch_grid_size=(24, 24), head_index=head)

# # 参数设定
# dim = 1024
# num_heads = dim // 64
# batch_size = 32
# seq_len = 10
#
# # 创建 MultiheadAttention 模块
# attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
#
# # 创建模拟数据
# query = torch.randn(batch_size, seq_len, dim)
# key_value = torch.randn(batch_size, seq_len*4, dim)
#
# # 测试 MultiheadAttention
# attn_output, _ = attn(query, key_value, value=key_value)
# print(f"attn_output shape: {attn_output.shape}")

