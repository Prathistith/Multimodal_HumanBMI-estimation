import torch

class CONFIG:

    vggface_path = "models/vggface.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 2
    train_drop_last = False

    body_model = "xception"
    body_input_shape = 512
    body_in_fts = 2048
    face_input_shape = 224

    input_dir = "/content/drive/MyDrive/DATA"
    output_dir = "/content/drive/MyDrive/DATA_OUT"
    checkpoints_dir = "/content/drive/MyDrive/checkpoints"
    obj_dir = output_dir


def get_config():
	return {key:value for key, value in CONFIG.__dict__.items() if not key.startswith('__')}

