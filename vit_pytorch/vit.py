import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 这是一个注意力机制的定义，它使用了 softmax 归一化、dropout 层和线性变换。下面是加上中文注释的代码：
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    # 这是一个注意力机制的定义，它使用了 softmax 归一化、dropout 层和线性变换。下面是加上中文注释的代码：
    def forward(self, x):
        x = self.norm(x)

        # 这是一个注意力机制的定义，它使用了 softmax 归一化、dropout 层和线性变换。下面是加上中文注释的代码：
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 这是一个注意力机制的定义，它使用了 softmax 归一化、dropout 层和线性变换。下面是加上中文注释的代码：
        attn = self.attend(dots)
        attn = self.dropout(attn)

        # 这是一个注意力机制的定义，它使用了 softmax 归一化、dropout 层和线性变换。下面是加上中文注释的代码：
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    # 这是一个 ViT 模型的定义，它使用了 Transformer 架构。下面是加上中文注释的代码：
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


# 这段代码定义了一个名为 ViT 的类，它是一种神经网络架构，其主要目的是实现图像分类任务。下面是加上注释后的代码：
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # 首先，我们需要检查图像的尺寸是否可以被补丁尺寸整除，如果可以，则计算补丁数量
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        # 接着，我们需要确保池化类型是 cls (类令牌) 或 mean (平均池化)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        # 接下来，我们需要定义positional embedding和类令牌，并将其添加到模型中。我们还需要定义转换器和MLP头，这些都是ViT模型的关键组成部分。
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

# 接下来，我们需要定义前向传播函数，该函数将输入图像转换为预测结果。在此过程中，我们将使用转换器将图像转换为表示，然后使用池化层将其转换为类向量。最后，我们将其送入MLP头中，以获得预测结果。
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
