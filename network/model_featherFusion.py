import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.nn as nn
from typing import List, Union

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        if len(x.shape)==2:
            x = x.unsqueeze(2)
            x = x.permute(2, 1, 0)
        # [batch_size, 3(feather), total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, 3(feather), 3 * total_embed_dim]
        # reshape: -> [batch_size, 3(feather), 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, 3(feather), embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, 3(feather), embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, 3(feather)]
        # @: multiply -> [batch_size, num_heads, 3(feather), 3(feather)]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, 3(feather), embed_dim_per_head]
        # transpose: -> [batch_size, 3(feather), num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, 3(feather), total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def build_mlp(in_dim: int,
              h_dim: Union[int, List],#int表示1层隐藏层，List表示2层及以上隐藏层
              out_dim: int = None,
              dropout_p: float = 0.2) -> nn.Sequential:
    """Builds an MLP.
    Parameters
    ----------
    in_dim: int,
        Input dimension of the MLP
    h_dim: int,
        Hidden layer dimension of the MLP
    out_dim: int, default None
        Output size of the MLP. If None, a Linear layer is returned, with ReLU
    dropout_p: float, default 0.2,
        Dropout probability
    """
    if isinstance(h_dim, int):#1层隐藏层也转成列表形式
        h_dim = [h_dim]

    sizes = [in_dim] + h_dim #输入层+隐藏层，连接
    mlp_size_tuple = list(zip(*(sizes[:-1], sizes[1:])))#去掉尾 去掉头  对齐头

    if isinstance(dropout_p, float):
        dropout_p = [dropout_p] * len(mlp_size_tuple)

    layers = []

    for idx, (prev_size, next_size) in enumerate(mlp_size_tuple):
        layers.append(nn.Linear(prev_size, next_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_p[idx]))

    if out_dim is not None:
        # layers.append(nn.Dropout(dropout_p[0]))
        layers.append(nn.Linear(sizes[-1], out_dim))

    return nn.Sequential(*layers)

class featherFusion(nn.Module):
    def __init__(self,num_classes,model_convNeXt,model_textCNN,model_GCN):
        super(featherFusion, self).__init__()
        self.flatten = nn.Flatten()
        self.model_convNeXt=model_convNeXt
        self.model_textCNN=model_textCNN
        self.model_GCN=model_GCN

        model_convNeXt.head=nn.Identity()
        model_textCNN.classifier=nn.Identity()
        model_GCN.classifier=nn.Identity()

        self.pool_k=5
        self.pool_o=1024
        self.dim=128
        
        joint_emb_size=self.pool_k*self.pool_o
        self.xnn=nn.Linear(self.dim,joint_emb_size)
        self.ynn=nn.Linear(self.dim,joint_emb_size)
        self.znn=nn.Linear(self.dim,joint_emb_size)

        self.attn = Attention(dim=self.pool_o, num_heads=4, qkv_bias=False, qk_scale=None,
                              attn_drop_ratio=0., proj_drop_ratio=0.)
        
        self.mlp = build_mlp(in_dim=self.pool_o, h_dim=[512,256,128,64], out_dim=num_classes, dropout_p=0.2)


    # def forward(self,x_image,x_code,x_graph):
    #     # print(X1.shape,X2.shape)

    #     x_image = self.model_convNeXt(x_image)
    #     x_code = self.model_textCNN(x_code)
    #     x_graph = self.model_GCN(x_graph)

    #     # x_fusion=torch.concat((x_image,x_code,x_graph),1)
    #     # MFB
    #     x_image=self.xnn(x_image)
    #     x_code=self.ynn(x_code)
    #     x_graph=self.znn(x_graph)

    #     out1 = torch.mul(x_image,x_code)
    #     out1 = out1.view(-1, 1, self.pool_o, self.pool_k) # batch, 1, o, k
    #     out1 = torch.squeeze(torch.sum(out1, 3))          # batch, o
    #     out1 = torch.sqrt(F.relu(out1)) - torch.sqrt(F.relu(-out1))    # Signed square root
        
    #     out2 = torch.mul(x_code,x_graph)
    #     out2 = out2.view(-1, 1, self.pool_o, self.pool_k) # batch, 1, o, k
    #     out2 = torch.squeeze(torch.sum(out2, 3))          # batch, o
    #     out2 = torch.sqrt(F.relu(out2)) - torch.sqrt(F.relu(-out2))    # Signed square root
        
    #     out3 = torch.mul(x_image,x_graph)
    #     out3 = out3.view(-1, 1, self.pool_o, self.pool_k) # batch, 1, o, k
    #     out3 = torch.squeeze(torch.sum(out3, 3))          # batch, o
    #     out3 = torch.sqrt(F.relu(out3)) - torch.sqrt(F.relu(-out3))    # Signed square root

    #     finalx = torch.stack([out1,out2,out3],dim=1)

    #     # out1 = F.normalize(out1)
    #     # out2 = F.normalize(out2)
    #     # out3 = F.normalize(out3)
        
    #     selfx = self.attn(finalx)
    #     selfx = torch.mean(selfx,dim=1)
    #     return self.mlp(selfx)

    def forward(self, x_image, x_code, x_graph, time_file=None):
        start_time = time.time()

        x_image = self.model_convNeXt(x_image)
        x_code = self.model_textCNN(x_code)
        x_graph = self.model_GCN(x_graph)

        x_image = self.xnn(x_image)
        x_code = self.ynn(x_code)
        x_graph = self.znn(x_graph)

        out1 = torch.mul(x_image, x_code)
        out1 = out1.view(-1, 1, self.pool_o, self.pool_k)
        out1 = torch.squeeze(torch.sum(out1, 3))
        out1 = torch.sqrt(F.relu(out1)) - torch.sqrt(F.relu(-out1))

        out2 = torch.mul(x_code, x_graph)
        out2 = out2.view(-1, 1, self.pool_o, self.pool_k)
        out2 = torch.squeeze(torch.sum(out2, 3))
        out2 = torch.sqrt(F.relu(out2)) - torch.sqrt(F.relu(-out2))

        out3 = torch.mul(x_image, x_graph)
        out3 = out3.view(-1, 1, self.pool_o, self.pool_k)
        out3 = torch.squeeze(torch.sum(out3, 3))
        out3 = torch.sqrt(F.relu(out3)) - torch.sqrt(F.relu(-out3))

        finalx = torch.stack([out1, out2, out3], dim=1)
        selfx = self.attn(finalx)
        selfx = torch.mean(selfx, dim=1)
        result = self.mlp(selfx)

        end_time = time.time()
        if time_file:
            with open(time_file, "a") as f:
                f.write(f"{end_time - start_time}\n")

        return result


def create_Fusion(num_classes: int,model_convNeXt,model_textCNN,model_GCN):
    model = featherFusion(num_classes=num_classes,model_convNeXt=model_convNeXt,model_textCNN=model_textCNN,model_GCN=model_GCN)
    return model
