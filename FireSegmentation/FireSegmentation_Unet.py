import threading
import random
import rasterio
import os
import numpy as np
import sys
import pandas as pd
from sklearn.utils import shuffle as shuffle_lists
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import copy
import joblib

MAX_PIXEL_VALUE = 65535 # 이미지 정규화를 위한 픽셀 최대값

class threadsafe_iter:
    """
    데이터 불러올떼, 호출 직렬화
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g

def get_img_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_762bands(path):
    img = rasterio.open(path).read((7,6,2)).transpose((1, 2, 0))    
    img = np.float32(img)/MAX_PIXEL_VALUE
    
    return img

def get_img_762bands_test(path):
    img = rasterio.open(path).read((7,6,2))  # 채널, 높이, 너비 순서
    img = np.float32(img) / MAX_PIXEL_VALUE
    # img = img.transpose((1, 2, 0))  # 이전 방식 (높이, 너비, 채널)
    # PyTorch가 기대하는 형식으로 이미지 차원 변경 불필요
    return img

    
def get_mask_arr(path):
    img = rasterio.open(path).read().transpose((1, 2, 0))
    seg = np.float32(img)
    return seg

@threadsafe_generator
def generator_from_lists(images_path, masks_path, batch_size=32, shuffle = True, random_state=None, image_mode='10bands'):
   
    images = []
    masks = []

    fopen_image = get_img_arr
    fopen_mask = get_mask_arr

    if image_mode == '762':
        fopen_image = get_img_762bands

    i = 0 
    # 데이터 shuffle
    while True:
        
        if shuffle:
            if random_state is None:
                images_path, masks_path = shuffle_lists(images_path, masks_path)
            else:
                images_path, masks_path = shuffle_lists(images_path, masks_path, random_state= random_state + i)
                i += 1 


        for img_path, mask_path in zip(images_path, masks_path):
            
            img = fopen_image(img_path)
            mask = fopen_mask(mask_path)
            images.append(img)
            masks.append(mask)

            if len(images) >= batch_size:
                yield (np.array(images), np.array(masks))
                images = []
                masks = []




def dice_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(y_true * y_pred, dim=[1,2,3])
    union = torch.sum(y_true, dim=[1,2,3]) + torch.sum(y_pred, dim=[1,2,3])
    dice = torch.mean((2. * intersection + smooth) / (union + smooth))
    return dice

def pixel_accuracy(y_true, y_pred):
    y_pred = torch.sigmoid(y_pred) > 0.5  # Sigmoid activation and thresholding
    correct = torch.eq(y_pred, y_true).sum().float()
    total = torch.numel(y_true)
    return correct / total

# Custom Dataset 클래스
class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        img = get_img_762bands(img_path)
        mask = get_mask_arr(mask_path)
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            
        return img, mask


# 사용할 데이터의 meta정보 가져오기

train_meta = pd.read_csv('./dataset/train_meta.csv')
test_meta = pd.read_csv('./dataset/test_meta.csv')


# 저장 이름
save_name = 'base_line'

N_FILTERS = 16 # 필터수 지정
N_CHANNELS = 3 # channel 지정
EPOCHS = 50 # 훈련 epoch 지정
BATCH_SIZE = 64 # batch size 지정
IMAGE_SIZE = (256, 256) # 이미지 크기 지정
MODEL_NAME = 'unet' # 모델 이름
RANDOM_STATE = 42 # seed 고정
INITIAL_EPOCH = 0 # 초기 epoch

# 데이터 위치
IMAGES_PATH = './dataset/train_img/'
MASKS_PATH = './dataset/train_mask/'

# 가중치 저장 위치
OUTPUT_DIR = './train_output/'
WORKERS = 4

# 조기종료
EARLY_STOP_PATIENCE = 5 

# 중간 가중치 저장 이름
CHECKPOINT_PERIOD = 5
CHECKPOINT_MODEL_NAME = 'checkpoint-{}-{}-epoch_{{epoch:02d}}.hdf5'.format(MODEL_NAME, save_name)
 
# 최종 가중치 저장 이름
FINAL_WEIGHTS_OUTPUT = 'model_{}_{}_final_weights.h5'.format(MODEL_NAME, save_name)

# 사용할 GPU 이름
CUDA_DEVICE = 0


# 저장 폴더 없으면 생성
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# PyTorch에서 GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device(f'cuda:{CUDA_DEVICE}') # GPU 디바이스 설정
    torch.cuda.set_device(device) # PyTorch에 사용할 기본 디바이스로 설정
    print(f'Using GPU: {torch.cuda.get_device_name(device)}')
else:
    device = torch.device('cpu')
    print('GPU is not available. Using CPU instead.')

# train : val = 8 : 2 나누기
x_tr, x_val = train_test_split(train_meta, test_size=0.2, random_state=RANDOM_STATE)
print(len(x_tr), len(x_val))

# 데이터 경로 설정
images_train = [os.path.join('./dataset/train_img/', image) for image in x_tr['train_img']]
masks_train = [os.path.join('./dataset/train_mask/', mask) for mask in x_tr['train_mask']]
images_validation = [os.path.join('./dataset/train_img/', image) for image in x_val['train_img']]
masks_validation = [os.path.join('./dataset/train_mask/', mask) for mask in x_val['train_mask']]

# 데이터 변환 정의
transform = transforms.Compose([transforms.ToTensor()])

# DataLoader 생성
train_loader = DataLoader(CustomDataset(images_train, masks_train, transform), batch_size=64, shuffle=True)
val_loader = DataLoader(CustomDataset(images_validation, masks_validation, transform), batch_size=64, shuffle=False)


def conv2d_block(in_channels, out_channels, kernel_size=3, batchnorm=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.ReLU(inplace=True)
    ]
    if batchnorm:
        layers.insert(1, nn.BatchNorm2d(out_channels))
    layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=16, dropout=0.1, batchnorm=True):
        super(UNet, self).__init__()
        
        # Contracting Path
        self.c1 = conv2d_block(n_channels, n_filters, batchnorm=batchnorm)
        self.p1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d1 = nn.Dropout(dropout)
        
        self.c2 = conv2d_block(n_filters, n_filters * 2, batchnorm=batchnorm)
        self.p2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d2 = nn.Dropout(dropout)

        # Expanding Path
        self.up1 = nn.ConvTranspose2d(n_filters * 2, n_filters, kernel_size=2, stride=2)
        self.c3 = conv2d_block(n_filters * 2, n_filters, batchnorm=batchnorm)

        self.final = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # Contracting Path
        c1 = self.c1(x)
        p1 = self.p1(c1)
        d1 = self.d1(p1)

        c2 = self.c2(d1)
        p2 = self.p2(c2)
        d2 = self.d2(p2)

        # Expanding Path
        up1 = self.up1(d2)
        
        # 업샘플링된 특성 맵의 크기를 c1의 크기에 맞게 조정
        up1 = F.interpolate(up1, size=c1.size()[2:], mode='bilinear', align_corners=False)
        
        merge1 = torch.cat([up1, c1], dim=1)
        c3 = self.c3(merge1)

        out = self.final(c3)
        return out

# 이미 정의된 UNet 모델 인스턴스 생성
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=N_CHANNELS, n_classes=1, n_filters=N_FILTERS, dropout=0.1, batchnorm=True).to(device)
# 옵티마이저 및 손실 함수 설정
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=1):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # 각 에포크는 학습 및 검증 단계를 갖습니다
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
                data_loader = train_loader
            else:
                model.eval()   # 모델을 평가 모드로 설정
                data_loader = val_loader
            
            running_loss = 0.0
            
            # 데이터를 반복
            for inputs, masks in data_loader:
                inputs = inputs.to(device)
                masks = masks.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)

                    # 학습 단계에서만 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
            
            epoch_loss = running_loss / len(data_loader.dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()

    print('Best val Loss: {:4f}'.format(best_loss))

    # 최적의 모델 가중치를 로드
    model.load_state_dict(best_model_wts)
    return model

# 모델 학습 및 검증
best_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS)

# 최적의 모델 가중치 저장
torch.save(best_model.state_dict(), os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT))

# 필요시 모델 가중치 로드
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, FINAL_WEIGHTS_OUTPUT)))
model.to(device)

import joblib

model.eval()  # 평가 모드
y_pred_dict = {}

with torch.no_grad():
    for i in test_meta['test_img']:
        img_path = os.path.join('./dataset/test_img/', i)
        img = get_img_762bands_test(img_path)
        img = torch.tensor(img).unsqueeze(0).to(device)  # 배치 차원 추가 및 디바이스 설정
        
        outputs = model(img)
        preds = torch.sigmoid(outputs).data > 0.5  # Sigmoid 활성화 및 임계값 적용
        preds = preds.cpu().numpy().astype(np.uint8)  # CPU로 이동 및 NumPy 배열로 변환
        
        y_pred_dict[i] = preds[0, 0, :, :]  # 배치 및 채널 차원 제거


# 예측 결과 저장
joblib.dump(y_pred_dict, './y_pred.pkl')
