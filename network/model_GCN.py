import dgl
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.data import MiniGCDataset
from dgl.nn.pytorch import GraphConv,GATConv
from dgl.nn.pytorch import SAGEConv
from sklearn.metrics import accuracy_score
 
class GCN(nn.Module):
    def __init__(self,num_classes):
        super(GCN, self).__init__()
        LEVELNUM=15
        self.conv1 = GraphConv(LEVELNUM, 256)  # 定义第一层图卷积
        self.conv2 = GraphConv(256, 128)  # 定义第二层图卷积
        self.classifier = nn.Linear(128, num_classes)   # 定义分类器
 
    def forward(self, g):
        # inputs = g.ndata['feat']
        # a=torch.sum(inputs,dim=0)
        h = F.relu(self.conv1(g,g.ndata['feat']))
        h = F.relu(self.conv2(g,h))
        g.ndata['h']=h
        h=dgl.mean_nodes(g,'h')
        logits = self.classifier(h)
        return logits


def create_GCN(num_classes: int):
    model = GCN(num_classes=num_classes)
    return model