import argparse
import joblib
import hashlib
import numpy as np
import pandas as pd
import torch,torchvision
import glob,os,sys,cv2,gc
from copy import deepcopy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings,time
warnings.filterwarnings("ignore")
from tqdm.notebook import tqdm
from config import get_config
from model import FinalModel
from data import OfflineDataset,OnlineDataset
from utils import IOStream,MetricMonitor,AlignDlib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None


def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, partition):
    ids = data.index.to_series()
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    if partition == "train":
      return data.loc[~in_test_set]
    elif partition == "test":
      return data.loc[in_test_set]


def check_checkpoints(args):
    ckpt = args['checkpoints_dir']
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)
    if not os.path.exists(os.path.join(ckpt,args['exp_name'])):
        os.makedirs(os.path.join(ckpt,args['exp_name']))
    if not os.path.exists(os.path.join(ckpt,args['exp_name'],'models')):
        os.makedirs(os.path.join(ckpt,args['exp_name'],'models'))


class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


def train(args,io):

    if args['meta_data_file'].endswith('xlsx'):
      data = pd.read_excel(os.path.join(args['input_dir'],args['meta_data_file']),sheet_name="Sheet1")
    elif args['meta_data_file'].endswith('csv'):
      data = pd.read_csv(os.path.join(args['input_dir'],args['meta_data_file']))

    if args['clean_ds']:
        io.cprint("Cleaning dataset...")
        data_csv = pd.DataFrame()
        # Including only reconstructed meshes for regression model.
        constructed_mesh_files = map(lambda x: os.path.basename(x).split("_")[1],glob.glob(os.path.join(args['output_dir'],"*.obj")))
        for filename in constructed_mesh_files:
          bool_jpg = data['file_name'].isin([f'{filename}.jpg'])
          bool_jpeg = data['file_name'].isin([f'{filename}.jpeg'])
          bool_png = data['file_name'].isin([f'{filename}.png'])
          bool_series = bool_jpg | bool_jpeg | bool_png
          data_csv = pd.concat([data_csv,data[bool_series]])
        
        # Only using alignable face images for regression model.
        detected_face_idxs = []
        aligner = AlignDlib("models/shape_predictor_68_face_landmarks.dat")
        for data_idx in data_csv.index:
            img_path = os.path.join(args['obj_dir'],data_csv.loc[data_idx,"file_name"])
            rgb_img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
            if aligner.align(args['face_input_shape'],rgb_img) is not None:
                detected_face_idxs.append(data_idx)
        data_csv = data_csv.loc[detected_face_idxs]
        data_csv.to_csv(os.path.join(args['input_dir'],"cleaned_Metadata.csv"))
        io.cprint(f"Cleaned data size: {data_csv.shape[0]}")
    else:
        data_csv = data
        io.cprint(f"Raw data size: {data_csv.shape[0]}")


    train_data = split_train_test_by_id(data_csv, args['test_ratio'], 'train')
    test_data = split_train_test_by_id(data_csv, args['test_ratio'], 'test')
    height_scaler = StandardScaler()
    age_scaler = StandardScaler()
    gender_encoder = OrdinalEncoder()

    train_data['height'] = height_scaler.fit_transform(train_data['height'].to_frame())
    train_data['age'] = age_scaler.fit_transform(train_data['age'].to_frame())
    train_data['sex'] = gender_encoder.fit_transform(train_data['sex'].to_frame())
    test_data['height'] = height_scaler.transform(test_data['height'].to_frame())
    test_data['age'] = age_scaler.transform(test_data['age'].to_frame())
    test_data['sex'] = gender_encoder.transform(test_data['sex'].to_frame())

    joblib.dump(age_scaler,os.path.join(args['checkpoints_dir'],args['exp_name'],'models','age_scaler.joblib'))
    joblib.dump(height_scaler,os.path.join(args['checkpoints_dir'],args['exp_name'],'models','height_scaler.joblib'))
    joblib.dump(gender_encoder,os.path.join(args['checkpoints_dir'],args['exp_name'],'models','gender_scaler.joblib'))

    parsed_dataset = {'train': train_data, 'test': test_data}
    if args['load_offline']:
        io.cprint("Initializing offline loading..")
        cleaned_ds_train = OfflineDataset(parsed_dataset,args,"train")
        cleaned_ds_test = OfflineDataset(parsed_dataset,args,"test")
    else:
        io.cprint("Initializing online loading..")
        cleaned_ds_train = OnlineDataset(parsed_dataset,args,"train")
        cleaned_ds_test = OnlineDataset(parsed_dataset,args,"test")

    train_ds = DataLoader(cleaned_ds_train,batch_size=args['batch_size'],shuffle=True,num_workers=args['num_workers'],\
    pin_memory=True,drop_last=args['train_drop_last'])
    test_ds = DataLoader(cleaned_ds_test,batch_size=args['test_batch_size'],shuffle=True,num_workers=args['num_workers'],pin_memory=True,drop_last=False)
    dataset_loaders = {'train': train_ds, 'test': test_ds}
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(FinalModel(args))
        model = model.to(args['device'])
    else:
        model = FinalModel(args).to(args['device'])
        
    scaler = torch.cuda.amp.GradScaler(enabled=args['use_amp'])
    huber_loss = torch.nn.SmoothL1Loss(beta=10)
    rmse_loss = RMSELoss()
    mae_loss = torch.nn.L1Loss()
    loss_choice_dict = {'mae': mae_loss, 'rmse' : rmse_loss, "huber" : huber_loss}
    criterion = loss_choice_dict[args['loss']].to(args['device'])

    if args['use_sgd']:
        print("Using SGD....")
        optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=1e-4)
    else:
        print("Using Adam....")
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, args['epochs'], eta_min=args['lr'])

    best_test_loss = float('inf')
    best_model_dict,batch_dscrp = None, None
    train_losses,test_losses = [],[]
    for epoch_count in range(args['epochs']):
        
        start_time = time.time()
        metric_monitor = MetricMonitor()
        metric_dict = metric_monitor.metrics

        for mode in ['train','test']:
            if mode == "train":
                model.train()
            elif mode == "test":
                model.eval()

            stream = tqdm(dataset_loaders[mode])
            for batch_number,inputs in enumerate(stream):
                points = torch.permute(inputs['points'],(0,2,1)).to(args['device'],non_blocking=True)
                cat_fts = inputs['cat_fts'].to(args['device'],non_blocking=True)
                body_imgs = inputs['body_imgs'].to(args['device'],non_blocking=True)
                face_imgs = inputs['face_imgs'].to(args['device'],non_blocking=True)
                target = inputs['targets'].to(args['device'],non_blocking=True)

                gc.collect()
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=args['use_amp']):
                    with torch.set_grad_enabled(mode == 'train'):
                        output = model(points,cat_fts,face_imgs,body_imgs)
                        loss = criterion(output,target)

                if mode == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                metric_monitor.update("Loss_" + mode, loss.detach().cpu().item())
                metric_monitor.update("Rmse_" + mode, rmse_loss(output.detach(),target.detach()).cpu().item())
                metric_monitor.update("Mae_" + mode, mae_loss(output.detach(),target.detach()).cpu().item())
                metric_monitor.update("Huber_" + mode, huber_loss(output.detach(),target.detach()).cpu().item())

                batch_summary = [f"{metric}: {metric_dict[metric]['avg']:.4f}" for metric in metric_dict.keys() if metric.endswith(mode)]
                batch_dscrp = f"{mode.capitalize()} - Epoch: {epoch_count+1}/{args['epochs']}, Batch: {batch_number+1}/{len(dataset_loaders[mode])} -> {' | '.join(batch_summary)}"
                stream.set_description(batch_dscrp)

            io.cprint(batch_dscrp)
            if mode == 'train':
                train_losses.append(float(batch_dscrp.split(": ")[3].split()[0]))
                scheduler.step()
            else:
                test_losses.append(float(batch_dscrp.split(": ")[3].split()[0]))

        if metric_dict["Loss_test"]['avg'] <= best_test_loss:
            best_test_loss = metric_dict["Loss_test"]['avg']
            best_model_dict = {"model": deepcopy(model.state_dict()), "optimizer": optimizer.state_dict(),\
            "test_ratio": args['test_ratio'],"scheduler": scheduler.state_dict(), "args_dict": args}
            io.cprint('Current Best: %.6f' % best_test_loss)
        
        if args['multi_modal'] == 'weighted_avg':
            if torch.cuda.device_count() > 1:
                io.cprint(f'Weights of mesh,face,body fts: {torch.squeeze(torch.softmax(model.module.weight,dim=0)).tolist()}')
            else:
                io.cprint(f'Weights of mesh,face,body fts: {torch.squeeze(torch.softmax(model.weight,dim=0)).tolist()}')
        
        io.cprint(f'Time taken for a epoch: {(time.time() - start_time):.2f} seconds\n')
        io.cprint("-" * 100)
    
    plt.plot(train_losses,label='train')
    plt.plot(test_losses, label='valid')
    plt.title("Training & Validation losses")
    plt.ylim([-1,150])
    plt.legend()
    torch.save(best_model_dict, os.path.join(args['checkpoints_dir'],args['exp_name'],'models',f'models_{best_test_loss:.6f}.tar'))
    plt.savefig(os.path.join(args['checkpoints_dir'],args['exp_name'],f"losses_{best_test_loss:.6f}.jpg"),dpi=150)
    

def test(args):

    if args['meta_data_file'].endswith('xlsx'):
      data_csv = pd.read_excel(os.path.join(args['input_dir'],args['meta_data_file']),sheet_name="Sheet1")
    elif args['meta_data_file'].endswith('csv'):
      data_csv = pd.read_csv(os.path.join(args['input_dir'],args['meta_data_file']))

    preds_io = IOStream(os.path.join(os.path.dirname(os.path.dirname(args['model_path'])), 'preds.log'))
    preds_io.cprint("Parsing input files...")
    model_load_dict = torch.load(args['model_path'],map_location=args['device'])
    test_data = split_train_test_by_id(data_csv, model_load_dict['test_ratio'], 'test')
    preds_io.cprint(f"Test data size: {test_data.shape[0]}")
    
    # Load pretrained model
    loaded_args = model_load_dict['args_dict']
    loaded_args['model_path'] = args['model_path']
    loaded_args['test_batch_size'] = args['test_batch_size']
    loaded_args['meta_data_file'] = args['meta_data_file']
    args = loaded_args
    preds_io.cprint(str(args))
    preds_io.cprint(f"{'-'*100}\n")

    age_scaler = joblib.load(os.path.join(os.path.dirname(args['model_path']),'age_scaler.joblib'))
    height_scaler = joblib.load(os.path.join(os.path.dirname(args['model_path']),'height_scaler.joblib'))
    gender_encoder = joblib.load(os.path.join(os.path.dirname(args['model_path']),'gender_scaler.joblib'))

    test_data['height'] = height_scaler.transform(test_data['height'].to_frame())
    test_data['age'] = age_scaler.transform(test_data['age'].to_frame())
    test_data['sex'] = gender_encoder.transform(test_data['sex'].to_frame())
    parsed_dataset = {'test': test_data}

    if args['load_offline']:
        preds_io.cprint("Initializing offline loading..")
        cleaned_ds_test = OfflineDataset(parsed_dataset,args,"test")
    else:
        preds_io.cprint("Initializing online loading..")
        cleaned_ds_test = OnlineDataset(parsed_dataset,args,"test")

    test_ds = DataLoader(cleaned_ds_test,batch_size=args['test_batch_size'],shuffle=False,num_workers=args['num_workers'],pin_memory=True,drop_last=False)
    dataset_loaders = {'test': test_ds}
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(FinalModel(args))
        model.load_state_dict(model_load_dict['model'])
        model = model.to(args['device'])
    else:
        model = FinalModel(args).to(args['device'])
        model.load_state_dict(model_load_dict['model'])
    model.eval()

    huber_loss = torch.nn.HuberLoss()
    rmse_loss = RMSELoss()
    mae_loss = torch.nn.L1Loss()

    outputs = []
    batch_dscrp = None
    metric_monitor = MetricMonitor()
    metric_dict = metric_monitor.metrics
    stream = tqdm(dataset_loaders['test'])


    for batch_number,inputs in enumerate(stream):
      points = torch.permute(inputs['points'],(0,2,1)).to(args['device'],non_blocking=True)
      cat_fts = inputs['cat_fts'].to(args['device'],non_blocking=True)
      body_imgs = inputs['body_imgs'].to(args['device'],non_blocking=True)
      face_imgs = inputs['face_imgs'].to(args['device'],non_blocking=True)
      target = inputs['targets'].to(args['device'],non_blocking=True)

      gc.collect()
      torch.cuda.empty_cache()

      with torch.set_grad_enabled(False):
        output = model(points,cat_fts,face_imgs,body_imgs)
        output_val = output.detach().cpu()
        outputs.append(output_val)

      rmse_val = rmse_loss(output.detach(),target.detach()).cpu().item()
      mae_val = mae_loss(output.detach(),target.detach()).cpu().item()
      huber_val = huber_loss(output.detach(),target.detach()).cpu().item()

      metric_monitor.update("Rmse_test" , rmse_val)
      metric_monitor.update("Mae_test", mae_val)
      metric_monitor.update("Huber_test", huber_val)

      batch_summary = [f"{metric}: {metric_dict[metric]['avg']}" for metric in metric_dict.keys() if metric.endswith('test')]
      batch_dscrp = f"Batch: {batch_number+1}/{len(dataset_loaders['test'])} -> {' | '.join(batch_summary)}"
      stream.set_description(batch_dscrp)
    
    if args['multi_modal'] == 'weighted_avg':
        if torch.cuda.device_count() > 1:
            preds_io.cprint(f'\nWeights of mesh,face,body fts: {torch.squeeze(torch.softmax(model.module.weight,dim=0)).tolist()}')
        else:
            preds_io.cprint(f'\nWeights of mesh,face,body fts: {torch.squeeze(torch.softmax(model.weight,dim=0)).tolist()}')
          
    preds_io.cprint(batch_dscrp)
    preds = torch.cat(outputs).squeeze().tolist()
    preds_io.cprint(f"Predictions are:\n{preds}")


if __name__ == "__main__":

    # ------------------------Training settings---------------------------------------------------

    parser = argparse.ArgumentParser(description='BMI-prediction')
    parser.add_argument('--exp_name', '-exp', type=str, required=True,
                        help='Name of the experiment')
    parser.add_argument('--model', dest='mesh_extractor', type=str, required=True, metavar='MODEL',
                        choices=['pointnet', 'dgcnn', 'gbnet'],
                        help='Model to use -> [pointnet, dgcnn, gbnet]')
    parser.add_argument('--loss', type=str, required=True, metavar='LOSS',
                        choices=['mae', 'rmse', 'huber'],
                        help='Loss to use -> [ MAE loss, RMSE loss, HUBER loss]')
    parser.add_argument('--multi_modal', type=str, required=True,
                        choices=['cat','avg','weighted_avg'],
                        help = 'Embedding aggregation methods - cat (concatenate), avg, weighted_avg')
    parser.add_argument('--test_ratio', '-tr', type=float, default=0.2, metavar='',
                        help='Test split ratio')
    parser.add_argument('--meta_data_file', type=str, required = True,
                        help='Path of meta data file')

    parser.add_argument('--load_offline', action='store_true',
                        help='Switches to offline data loading')
    parser.add_argument('--clean_ds', action='store_true',
                        help='Toggles b/w using raw data & processed data')
    parser.add_argument('--use_amp', '-amp', action='store_true',
                        help='Switches between fp16/32 training')
    parser.add_argument('--use_sgd', action='store_true',
                        help='Uses SGD optimizer over Adam')
    parser.add_argument('--eval', action = 'store_true',
                        help='Switches to evaluation mode')

    parser.add_argument('--batch_size', '-bs',type=int, default=32, metavar = '',
                        help='Size of batch')
    parser.add_argument('--test_batch_size', '-tbs', type=int, default=16, metavar='',
                        help='Size of validation batch')
    parser.add_argument('--epochs', type=int, required = True, metavar='EPOCHS',
                        help='number of episodes to train')
    parser.add_argument('--lr', type=float, required = True, metavar='LR',
                        help='learning rate (default: 0.001. Try - 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='SGD-M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--num_points', '-points', type=int, default=1024, metavar='',
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='DRP',
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='ED',
                        help='Dimension of embeddings for 3d feature extraction')
    parser.add_argument('--k', type=int, default=20, metavar='K',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='Seed value for reproducibility.')
    parser.add_argument('--model_path', type=str, default='', metavar='MP',
                        help='Pretrained model path')

    model_args = vars(parser.parse_args())
    path_args = get_config()
    args = {**path_args,**model_args}
    torch.manual_seed(args['seed'])

    if args['device'] == 'cuda':
        if torch.cuda.device_count() >1:
            device_use = torch.cuda.device_count()
        else:
            device_use = torch.cuda.current_device()+1
        print('Using GPU : '+str(device_use)+' from '+str(torch.cuda.device_count())+' device(s)..')
        torch.cuda.manual_seed(args['seed'])
    else:
        print('Using CPU...')

    if not args['eval']:
        check_checkpoints(args)
        io = IOStream(os.path.join(args['checkpoints_dir'], args['exp_name'], 'run.log'))
        io.cprint(str(args))
        train(args, io)
    else:
        test(args)
