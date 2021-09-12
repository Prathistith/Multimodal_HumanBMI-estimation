import h5py
import torch
import trimesh
import numpy as np
import cv2,os,sys,glob
from torchvision import transforms
from torch.utils.data import Dataset
from utils import AlignDlib

def crop_image(img, rect,args):
    x, y, w, h = rect
    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1]-(x+w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0]-(y+h)) if y + h >= img.shape[0] else 0

    if img.shape[2] == 4:
        color = [0, 0, 0, 0]
    else:
        color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top
    new_img = new_img[y:(y+h),x:(x+w),:]
    return cv2.resize(new_img,(args['body_input_shape'],args['body_input_shape']))


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def parse_dataset(data,args,aligner):
    points_fts,cat_fts,face_fts,body_fts,targets = [],[],[],[],[]
    for idx in data.index:
        fname = os.path.splitext(data.loc[idx,"file_name"])[0]
        file_name = os.path.join(args['obj_dir'],f'result_{fname}_256.obj')

        #print("Processing: ",os.path.basename(file_name))
        try:
          mesh = trimesh.load(file_name)
        except:
          print(f"Unable to read {file_name} -> skipping...")
          continue
        points = mesh.sample(args['num_points'])
        point_cloud = (points/points.max(axis=0)) * mesh.extents

        img_path = os.path.join(args['obj_dir'],data.loc[idx,"file_name"])
        rgb_img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        aligned_img = aligner.align(args['face_input_shape'],rgb_img)
        if aligned_img is None:
          continue
        face_img = cv2.cvtColor(aligned_img,cv2.COLOR_RGB2BGR).astype(np.float32)
        face_img -= np.array([129.1863, 104.7624, 93.5940])

        rect_path = os.path.join(args['obj_dir'],(fname + "_rect.txt"))
        rect = np.loadtxt(rect_path,dtype=np.int32)
        if rect.shape[0] != 4:
            rect = rect[0]
        body_img = crop_image(rgb_img,rect,args)
        label = data.loc[idx,"weight"]

        targets.append([[label]])
        points_fts.append([point_cloud])
        cat_fts.append([[data.loc[idx,"sex"],data.loc[idx,"age"],data.loc[idx,"height"]]])
        face_fts.append([face_img])
        body_fts.append([body_img])

    fts_list = [points_fts,cat_fts,face_fts,body_fts,targets]
    for idx,item in enumerate(fts_list):
        fts_list[idx] = np.concatenate(item,axis=0).astype(np.float32)
    return (*fts_list,)


class OfflineDataset(Dataset):
    def __init__(self,dataset,args,partition='train'):
        self.aligner = AlignDlib("models/shape_predictor_68_face_landmarks.dat")
        self.data = parse_dataset(dataset[partition],args,self.aligner)
        self.partition = partition
        self.body_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.face_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])

    def __getitem__(self, idx):
        points,cat_fts,face_imgs,body_imgs,targets = self.data
        points_out,cat_fts_out,face_imgs_out,body_imgs_out,targets_out = points[idx],cat_fts[idx],face_imgs[idx],body_imgs[idx],targets[idx]
        body_imgs_out = self.body_transform(body_imgs_out)
        face_imgs_out = self.face_transform(face_imgs_out)

        if self.partition == 'train':
            points_out = translate_pointcloud(points_out)
            np.random.shuffle(points_out)
        return {'points': points_out, 'cat_fts': cat_fts_out, 'face_imgs': face_imgs_out,'body_imgs': body_imgs_out, 'targets': targets_out}

    def __len__(self):
        return self.data[0].shape[0]



class OnlineDataset(Dataset):
    def __init__(self,dataset,args,partition='train'):
        self.aligner = AlignDlib("models/shape_predictor_68_face_landmarks.dat")
        self.args = args
        self.data = dataset[partition]
        self.partition = partition
        self.body_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
        self.face_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])

    def __getitem__(self, idx):

        file_name_idx = self.data.columns.get_loc("file_name")
        sex_idx = self.data.columns.get_loc("sex")
        age_idx = self.data.columns.get_loc("age")
        height_idx = self.data.columns.get_loc("height")
        fname = os.path.splitext(self.data.iloc[idx,file_name_idx])[0]


        obj_file_name = os.path.join(self.args['obj_dir'],f'result_{fname}_256.obj')
        #print("Processing: ",os.path.basename(obj_file_name))
        mesh = trimesh.load(obj_file_name)
        points = mesh.sample(self.args['num_points'])
        point_cloud = (points/points.max(axis=0)) * mesh.extents


        img_path = os.path.join(self.args['obj_dir'],self.data.iloc[idx,file_name_idx])
        rgb_img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        face_img = cv2.cvtColor(self.aligner.align(self.args['face_input_shape'],rgb_img),cv2.COLOR_RGB2BGR).astype(np.float32)
        face_img -= np.array([129.1863, 104.7624, 93.5940])


        rect_path = os.path.join(self.args['obj_dir'],(fname + "_rect.txt"))
        rect = np.loadtxt(rect_path,dtype=np.int32)
        if rect.shape[0] != 4:
            rect = rect[0]
        body_img = crop_image(rgb_img,rect,self.args)
        label = self.data.iloc[idx,self.data.columns.get_loc("weight")]

        body_img = self.body_transform(body_img)
        face_img = self.face_transform(face_img)

        if self.partition == 'train':
            points_cloud = translate_pointcloud(point_cloud)
            np.random.shuffle(point_cloud)


        out_dict = {'points': torch.as_tensor(point_cloud,dtype=torch.float32),
        'cat_fts': torch.as_tensor([self.data.iloc[idx,sex_idx],self.data.iloc[idx,age_idx],self.data.iloc[idx,height_idx]],dtype=torch.float32),
        'face_imgs': torch.as_tensor(face_img,dtype=torch.float32),
        'body_imgs': torch.as_tensor(body_img,dtype=torch.float32),
        'targets': torch.as_tensor([label],dtype=torch.float32)
        }

        return out_dict

    def __len__(self):
        return self.data.shape[0]
