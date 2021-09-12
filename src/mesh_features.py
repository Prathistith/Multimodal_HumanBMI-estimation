import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


#--------------------------------------------------------------------POINTNET---------------------------------------------------------------------------

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNet(nn.Module):
    def __init__(self, feature_transform=False):
        super(PointNet, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        return x #trans, trans_feat

def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


#---------------------------------------------------------------------DGCNN---------------------------------------------------------------------------

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args['k']

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args['emb_dims'])

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args['emb_dims'], kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args['emb_dims']*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        #self.dp1 = nn.Dropout(p=args['dropout'])
        #self.linear2 = nn.Linear(512, 256)
        #self.bn7 = nn.BatchNorm1d(256)
        #self.dp2 = nn.Dropout(p=args['dropout'])
        #self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        #x = self.dp1(x)
        #x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        #x = self.dp2(x)
        #x = self.linear3(x)
        return x


#---------------------------------------------------------------------GBNET---------------------------------------------------------------------------

def geometric_point_descriptor(x, k=3, idx=None):
    # x: B,3,N
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([1])) # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.cuda.LongTensor([2])) # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st-org_x
    edge2 = neighbor_2nd-org_x
    normals = torch.cross(edge1, edge2, dim=1) # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True) # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True) # B,1,N

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1) # B,14,N

    return new_pts


class CAA_Module(nn.Module):
    """ Channel-wise Affinity Attention module"""
    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(1024//8)
        self.bn2 = nn.BatchNorm1d(1024//8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024//8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=1024, out_channels=1024//8, kernel_size=1, bias=False),
                                        self.bn2,
                                        nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X N )
            returns :
                out : output feature maps( B X C X N )
        """

        # Compact Channel-wise Comparator block
        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)

        # Channel Affinity Estimator block
        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat)-similarity_mat
        affinity_mat = self.softmax(affinity_mat)

        proj_value = self.value_conv(x)
        out = torch.bmm(affinity_mat, proj_value)
        # residual connection with a learnable weight
        out = self.alpha*out + x
        return out


class ABEM_Module(nn.Module):
    """ Attentional Back-projection Edge Features Module (ABEM)"""
    def __init__(self, in_dim, out_dim, k):
        super(ABEM_Module, self).__init__()

        self.k = k
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn2 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Sequential(nn.Conv2d(out_dim, in_dim, kernel_size=[1,self.k], bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa1 = CAA_Module(out_dim)

        self.bn4 = nn.BatchNorm2d(out_dim)
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim*2, out_dim, kernel_size=[1,self.k], bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa2 = CAA_Module(out_dim)

    def forward(self,x):
        # Prominent Feature Encoding
        x1 = x # input
        input_edge = get_graph_feature(x, k=self.k)
        x = self.conv1(input_edge)
        x2 = x # EdgeConv for input

        x = self.conv2(x) # LFC
        x = torch.squeeze(x, -1)
        x3 = x # Back-projection signal

        delta = x3 - x1 # Error signal

        x = get_graph_feature(delta, k=self.k)  # EdgeConv for Error signal
        x = self.conv3(x)
        x4 = x

        x = x2 + x4 # Attentional feedback
        x = x.max(dim=-1, keepdim=False)[0]
        x = self.caa1(x) # B,out_dim,N

        # Fine-grained Feature Encoding
        x_local = self.conv4(input_edge)
        x_local = torch.squeeze(x_local, -1)
        x_local = self.caa2(x_local) # B,out_dim,N

        return x, x_local


class GBNet(nn.Module):
    def __init__(self, args):
        super(GBNet, self).__init__()
        self.args = args
        self.k = args['k']

        self.abem1 = ABEM_Module(14, 64, self.k)
        self.abem2 = ABEM_Module(64, 64, self.k)
        self.abem3 = ABEM_Module(64, 128, self.k)
        self.abem4 = ABEM_Module(128, 256, self.k)

        self.bn = nn.BatchNorm1d(args['emb_dims'])
        self.conv = nn.Sequential(nn.Conv1d(1024, args['emb_dims'], kernel_size=1, bias=False),
                                   self.bn,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa = CAA_Module(1024)

        self.linear1 = nn.Linear(args['emb_dims'] * 2, 512, bias=False)
        self.bn_linear1 = nn.BatchNorm1d(512)
        #self.dp1 = nn.Dropout(p=args['dropout'])
        #self.linear2 = nn.Linear(512, 256)
        #self.bn_linear2 = nn.BatchNorm1d(256)
        #self.dp2 = nn.Dropout(p=args['dropout'])
        #self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        # x: B,3,N
        batch_size = x.size(0)

        # Geometric Point Descriptor:
        x = geometric_point_descriptor(x) # B,14,N

        # 1st Attentional Back-projection Edge Features Module (ABEM):
        x1, x1_local = self.abem1(x)

        # 2nd Attentional Back-projection Edge Features Module (ABEM):
        x2, x2_local = self.abem2(x1)

        # 3rd Attentional Back-projection Edge Features Module (ABEM):
        x3, x3_local = self.abem3(x2)

        # 4th Attentional Back-projection Edge Features Module (ABEM):
        x4, x4_local = self.abem4(x3)

        # Concatenate both prominent and fine-grained outputs of 4 ABEMs:
        x = torch.cat((x1, x1_local, x2, x2_local, x3, x3_local, x4, x4_local), dim=1)  # B,(64+64+128+256)x2,N
        x = self.conv(x)
        x = self.caa(x) # B,1024,N

        # global embedding
        global_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        global_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((global_max, global_avg), 1)  # B,2048

        # FC layers with dropout
        x = F.leaky_relu(self.bn_linear1(self.linear1(x)), negative_slope=0.2)  # B,512
        #x = self.dp1(x)
        #x = F.leaky_relu(self.bn_linear2(self.linear2(x)), negative_slope=0.2)  # B,256
        #x = self.dp2(x)
        #x = self.linear3(x)  # B,C
        return x
