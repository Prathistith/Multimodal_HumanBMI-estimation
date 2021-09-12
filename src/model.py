import numpy as np
import torch,os
import torch.nn as nn
import torch.nn.functional as F
from img_features import FaceFeatures,BodyFeatures
from mesh_features import PointNet,DGCNN,GBNet


class FinalModel(nn.Module):
	def __init__(self,args):
		super(FinalModel, self).__init__()

		if args['mesh_extractor'] == "pointnet":
			self.mesh_extractor = PointNet(args)
		elif args['mesh_extractor'] == "dgcnn":
			self.mesh_extractor = DGCNN(args)
		elif args['mesh_extractor'] == "gbnet":
			self.mesh_extractor = GBNet(args)

		self.args = args
		self.face_extractor = FaceFeatures(args)
		self.body_extractor = BodyFeatures(args)

		self.linear1 = nn.Linear((512 * 3)+3, 1024)
		self.bn1 = nn.BatchNorm1d(1024)
		self.dp1 = nn.Dropout(p=args['dropout'])

		self.linear2 = nn.Linear(1024, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.dp2 = nn.Dropout(p=args['dropout'])

		self.linear3 = nn.Linear(1024, 512)
		self.bn3 = nn.BatchNorm1d(512)
		self.dp3 = nn.Dropout(p=args['dropout'])

		self.linear4 = nn.Linear(512, 512)
		self.bn4 = nn.BatchNorm1d(512)
		self.dp4 = nn.Dropout(p=args['dropout'])

		self.linear5 = nn.Linear(512, 256)
		self.bn5 = nn.BatchNorm1d(256)
		self.linear6 = nn.Linear(256, 1)

		self.linear7 = nn.Linear(512+3, 512)
		self.bn7 = nn.BatchNorm1d(512)
		self.dp7 = nn.Dropout(p=args['dropout'])
		self.weight = nn.Parameter(torch.nn.init.uniform_(torch.empty(3,1,1)))


	def forward(self,points,cat_fts,face_fts,body_fts):

		x1 = self.mesh_extractor(points)
		x2 = self.face_extractor(face_fts)
		x3 = self.body_extractor(body_fts)

		if self.args['multi_modal'] == 'cat':
			x = torch.cat((x1,x2,x3,cat_fts),1)

			x = F.relu(self.bn1(self.linear1(x.float())))
			x = self.dp1(x)

			x = F.relu(self.bn2(self.linear2(x)))
			x = self.dp2(x)

			x = F.relu(self.bn3(self.linear3(x)))
			x = self.dp3(x)

		elif self.args['multi_modal'] == 'avg':
			x = torch.stack((x1,x2,x3))
			x = torch.mean(x,0)
			x = torch.cat((x,cat_fts),1)
			x = F.relu(self.bn7(self.linear7(x)))

		elif self.args['multi_modal'] == 'weighted_avg':
			x = torch.stack((x1,x2,x3))
			x = (torch.softmax(self.weight,dim=0) * x)
			x = torch.mean(x,0)
			x = torch.cat((x,cat_fts),1)
			x = F.relu(self.bn7(self.linear7(x)))

		x = F.relu(self.bn4(self.linear4(x)))
		x = self.dp4(x)

		x = F.relu(self.bn5(self.linear5(x)))
		x = F.relu(self.linear6(x))

		return x
