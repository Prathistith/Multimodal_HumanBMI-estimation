import torch
import random
import argparse
import cv2,os,gc
import torchvision
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import trimesh,sys,glob,shutil
from utils import download_file
import matplotlib.pyplot as plt
import torchvision.transforms as T


class RCNN:
  def __init__(self,img_path,output_dir,resize_ratio,threshold=0.8, mask_threshold=0.1, rect_th=3, text_size=3, text_th=3):
    self.img_path = img_path
    self.threshold = threshold
    self.mask_threshold = mask_threshold
    self.rect_th = rect_th
    self.text_size = text_size
    self.text_th = text_th
    self.out_dir = output_dir
    self.resize_ratio = resize_ratio

  def get_prediction(self,model,img_path,threshold,mask_threshold):
    COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(0,0),img,self.resize_ratio,self.resize_ratio)
    img = Image.fromarray(img)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])

    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
    masks = (pred[0]['masks']>mask_threshold).squeeze().detach().cpu().numpy()
    if masks.ndim < 3:
      masks = masks[np.newaxis]
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]

    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class,pred

  def random_colour_masks(self,image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],\
    [80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

  def get_final_mask(self,filepath,mask,whitebg=True):
    input_img = cv2.cvtColor(cv2.imread(filepath)[...,:3],cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img,(0,0),input_img,self.resize_ratio,self.resize_ratio)
    if whitebg:
      fg = cv2.bitwise_and(input_img,input_img,mask = mask)
      bg = np.full((fg.shape[0],fg.shape[1],3),255,dtype=np.uint8)
      inv_mask = cv2.bitwise_not(mask)
      bg = cv2.bitwise_or(bg,bg,mask=inv_mask)
      result = cv2.bitwise_or(fg,bg)
    else:
      result = cv2.bitwise_and(input_img,input_img,mask=mask)
    return result

  def instance_segmentation(self,img_path,threshold,mask_threshold,rect_th,text_size,text_th):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    masks, boxes, pred_cls,preds = self.get_prediction(model,img_path,threshold,mask_threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(0,0),img,self.resize_ratio,self.resize_ratio)
    people_dict = {"masks": [], "boxes": []}

    for i in range(len(masks)):
      if pred_cls[i] == "person":
        rgb_mask = self.random_colour_masks(masks[i])
        #print(img.shape,rgb_mask.shape,masks[i].shape)
        img = cv2.addWeighted(img, 1, rgb_mask, 0.7, 0)
        cv2.rectangle(img,(int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]),int(boxes[i][1][1])), color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
        people_dict["masks"].append(masks[i])
        people_dict["boxes"].append({"x1_y1": boxes[i][0], "x2_y2": boxes[i][1]})
    return preds,people_dict,img

  def run(self,crop=False,save_mask=True):
    gc.collect()
    torch.cuda.empty_cache()
    preds,people_preds,color_masked_img = self.instance_segmentation(self.img_path,self.threshold,self.mask_threshold,\
                                                                    self.rect_th,self.text_size,self.text_th)
    person_mask = people_preds["masks"][0].astype(np.uint8) * 255
    final_masked_img = self.get_final_mask(self.img_path,person_mask,False)
    x1,y1 = np.ceil(people_preds["boxes"][0]["x1_y1"])
    x2,y2 = np.ceil(people_preds["boxes"][0]["x2_y2"])
    x1,x2,y1,y2 = int(x1),int(x2),int(y1),int(y2)
    height,width = abs(y1-y2), abs(x1 - x2)
    dst_path = os.path.join(self.out_dir,os.path.basename(self.img_path))

    if save_mask == True:
      if crop == True:
        final_masked_img = final_masked_img[min(y1,y2):max(y1,y2),min(x1,x2):max(x1,x2)]
      final_pil_img = Image.fromarray(final_masked_img)
      final_pil_img.save(dst_path)
    else:
      shutil.copy(self.img_path,dst_path)
    return final_masked_img,height,width,dst_path



class KeyPointDetection:
  def __init__(self,img_dir):
    self.img_dir = img_dir
    self.device = 'cuda'
    sys.path.append('src/pose_estimation')

  def get_rect(self,net, img_dir, height_size):
    from pose_estimation.modules.keypoints import extract_keypoints, group_keypoints
    from pose_estimation.modules.pose import Pose, track_poses
    import demo

    net = net.eval()
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 33

    print("\nDetecting keypoints....")
    all_img_paths = glob.glob(os.path.join(img_dir,"*.jpg"))+glob.glob(os.path.join(img_dir,"*.jpeg"))+glob.glob(os.path.join(img_dir,"*.png"))
    for image in all_img_paths:
        print(image)
        skip = False
        rect_path = image.replace('.%s' % (image.split('.')[-1]), '_rect.txt')
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = demo.infer_fast(net, img, height_size, stride, upsample_ratio, cpu=False)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []

        rects = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            valid_keypoints = []
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
            valid_keypoints = np.array(valid_keypoints)

            if pose_entries[n][10] != -1.0 or pose_entries[n][13] != -1.0:
              pmin = valid_keypoints.min(0)
              pmax = valid_keypoints.max(0)

              center = (0.5 * (pmax[:2] + pmin[:2])).astype(np.int)
              radius = int(0.65 * max(pmax[0]-pmin[0], pmax[1]-pmin[1]))

            elif pose_entries[n][10] == -1.0 and pose_entries[n][13] == -1.0 and pose_entries[n][8] != -1.0 and pose_entries[n][11] != -1.0:
              # if leg is missing, use pelvis to get cropping
              print(f"Using pelvis to get cropping - > {image} -- skipping")
              skip = True
              continue
              center = (0.5 * (pose_keypoints[8] + pose_keypoints[11])).astype(np.int)
              radius = int(1.45*np.sqrt(((center[None,:] - valid_keypoints)**2).sum(1)).max(0))
              center[1] += int(0.05*radius)

            else:
              print(f"Required key-points not found - > {image} -- skipping")
              skip = True
              center = np.array([img.shape[1]//2,img.shape[0]//2])
              radius = max(img.shape[1]//2,img.shape[0]//2)

            x1 = center[0] - radius
            y1 = center[1] - radius
            rects.append([x1, y1, 2*radius, 2*radius])

        if not skip:
            np.savetxt(rect_path, np.array(rects), fmt='%d')

  def run(self):
    from pose_estimation.models.with_mobilenet import PoseEstimationWithMobileNet
    from pose_estimation.modules.load_state import load_state
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load('src/pose_estimation/checkpoint_iter_370000.pth', map_location=self.device)
    load_state(net, checkpoint)
    self.get_rect(net.cuda(), self.img_dir, 512)


def clean_mesh(in_mesh_dir,out_mesh_dir):
    for obj_file in glob.glob(os.path.join(in_mesh_dir,"*.obj")):
        meshes = trimesh.load_mesh(obj_file)
        meshes_list = meshes.split()
        max_vol_mesh = None

        if len(meshes_list) > 0:
            volumes_list = list(map(lambda x: x.volume,meshes_list))
            max_vol_idx = volumes_list.index(max(volumes_list))
            max_vol_mesh = meshes_list[max_vol_idx]
            print("Volume :",max_vol_mesh.volume)
            max_vol_mesh.export(os.path.join(out_mesh_dir,os.path.basename(obj_file)))
        else:
          print(f"Mesh not found - > {obj_file}")



if __name__ == '__main__':
    
    # ------------------------------Pre-processing argument parser------------------------------
    
    parser = argparse.ArgumentParser(description='BMI-prediction-preprocessing')
    parser.add_argument('--input_dir', '-i', type=str, required=True, metavar='',
                        help='Input directory')
    parser.add_argument('--output_dir', '-o', type=str, required=True, metavar='',
                        help='Output directory')
    parser.add_argument('--resize_ratio','-rr', type=float, required=True, metavar='',
                        help='Resize ratio for resizing imgs while masking')
    args = vars(parser.parse_args())
    
    inputs_dir = args['input_dir']
    outputs_dir = args['output_dir']
    resize_ratio = args['resize_ratio']
    
    all_img_paths = glob.glob(os.path.join(inputs_dir,"*.jpg"))+glob.glob(os.path.join(inputs_dir,"*.jpeg"))+glob.glob(os.path.join(inputs_dir,"*.png"))
    print("Masking images...")

    for in_img_path in all_img_paths:
        print(in_img_path)
        gc.collect()
        torch.cuda.empty_cache()
        try:
            rcnn = RCNN(in_img_path,outputs_dir,resize_ratio,mask_threshold=0.01)
            rcnn.run(True,True)
        except Exception as e:
            print(f"Error masking -> {in_img_path}")
        
    print("-"*100)
    kpd = KeyPointDetection(outputs_dir)
    kpd.run()

    os.chdir("src/pifuhd/")
    sys.path.append('src/pifuhd')

    os.makedirs("checkpoints",exist_ok=True)
    os.chdir("checkpoints")
    download_file("https://dl.fbaipublicfiles.com/pifuhd/checkpoints/pifuhd.pt","pifuhd.pt")
    os.chdir("..")

    #os.system("sh ./scripts/download_trained_model.sh")
    gc.collect()
    torch.cuda.empty_cache()
    os.system(f"python -m apps.simple_test -r 256 --use_rect -i {os.path.relpath(outputs_dir)}")

    clean_mesh("results/pifuhd_final/recon/", os.path.relpath(outputs_dir))
    shutil.rmtree("results/pifuhd_final/recon/")
