# https://github.com/AshkanGanj/LR-Net
import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)

CFG = {
    'IMG_SIZE': 224,
    'EPOCHS': 3,
    'LEARNING_RATE': 3e-4,
    'BATCH_SIZE': 8,
    'SEED': 41
}

def seed_everything(seed):
    random.seed(seed) ##random module의 시드 고정
    os.environ['PYTHONHASHSEED'] = str(seed) #해시 함수의 랜덤성 제어, 자료구조 실행할 때 동일한 순서 고정
    np.random.seed(seed) #numpy 랜덤 숫자 일정
    torch.manual_seed(seed) # torch라이브러리에서 cpu 텐서 생성 랜덤 시드 고정
    torch.cuda.manual_seed(seed) # cuda의 gpu텐서에 대한 시드 고정
    torch.backends.cudnn.deterministic = True # 백엔드가 결정적 알고리즘만 사용하도록 고정 
    torch.backends.cudnn.benchmark = True # CuDNN이 여러 내부 휴리스틱을 사용하여 가장 빠른 알고리즘 동적으로 찾도록 설정

df = pd.read_csv('./train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=CFG['SEED'])


le = preprocessing.LabelEncoder() # 라벨인코딩 /라벨(목표 변수)를 정수로 인코딩
# train, label의 라벨인코딩 과정 진행
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])

class CustomDataset(Dataset):
## 파일 경로와 라벨을 받아, 데이터를 로드하고 전처리하는 데이터셋 생성성
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        # 이미지 읽어오기
        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image = image)['image']

        # 라벨이 있다면 이미지와 함께 반환
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        # 라벨이 없다면 이미지만 반환환
        else:
            return image
    
    def __len__(self):
        return len(self.img_path_list)
    

# Compose는 여러 변환을 연속적으로적용할 수 있게 해주는 함수. (IMG 사이즈 224로 설정되어 있음)
# 이미지 크기조정, 정구화, 텐서로 변환 포함.
'''
Normalize(mean=0.485, 0.456, 0.406값은 각 채널별 평균)
std=(0.229, 0.224, 0.225 값은 각 채널별 표준편차)
max_pixel_value: 이미지의 최대 픽셀 값 (8비트의 경우 255가 최대값)
always_apply= Ture: 변환이 데이터셋의 모든 이미지에 대해 항상 적용.
p: 변환이 적용될 확률: (0~1 사이)
대부분의 경우 always_apply=True로 하고 p를 조절해서 사용 
'''
train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()])

test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()])

## train데이터셋 설정 및 불러오기
train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

## val 데이터셋 설정 및 불러오기
val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

# MK_block definition (as you provided)
class MK_block(nn.Module):
    def __init__(self, input_channels):
        super(MK_block, self).__init__()
        self.conv3 = nn.Conv2d(input_channels, 48, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(input_channels, 24, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(input_channels, 12, kernel_size=7, padding=3)
        # Adjusted batch norm channels to sum of output channels from conv layers
        self.batch_norm = nn.BatchNorm2d(48 + 24 + 12)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.relu(self.conv3(x))
        x2 = self.relu(self.conv5(x))
        x3 = self.relu(self.conv7(x))
        out = torch.cat((x1, x2, x3), 1)
        out = self.batch_norm(out)
        return out

# Net definition (replacing BaseModel with this)
class Net(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(Net, self).__init__()
        # Start with 3 channels as input for the first block
        self.block1 = MK_block(3)
        # Output from the first block will have 48 + 24 + 12 = 84 channels
        self.block2 = MK_block(84)
        # The second block also outputs 84 channels
        self.block3 = MK_block(84)
        self.flatten = nn.Flatten()
        # Calculate the output size after flattening; adjust based on your actual network and image size
        output_size = 84 * 224 * 224  # This needs to be calculated based on your actual output dimensions
        self.fc1 = nn.Linear(output_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.dropout(self.fc3(x))
        x = self.fc4(x)
        return x


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device) # 모델을 해당 디바이스로 옮김(cpu, gpu)
    criterion = nn.CrossEntropyLoss().to(device) # 손실함수 정의하고 해당 device로 옮김
    
    # 성능 기록 초기화
    best_score = 0
    best_model = None
    
    # 설정한 하이퍼파라미터의 epochs만큼 반복
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train() #모델을 훈련모드로 설정
        train_loss = []

        # 반복을 통해서 배치 단위로 이미지와 라벨을 가져옴
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device) # 이미지를 실수형으로 변경한 후 device로 올림
            labels = labels.long().to(device)  # 데이터 타입 long으로 변경한 후 device로 올림 (int로 변경하였을 때, error 발생했음)
            
            optimizer.zero_grad() # 이전 그레디언트가 누적될 가능성이 있으니 초기화
            
            output = model(imgs) # 모델의 이미지를 입력하여 출력을 얻음
            loss = criterion(output, labels) # 손실 함수를 통해 손실 값을 계산함.
            
            loss.backward() # 손실에 대한 그레디언트 계산
            optimizer.step() # 옵티마이저를 통해 모델의 가중치 업데이트
            
            train_loss.append(loss.item()) # loss.item()은 현재 배치에 대한 손실 값을 파이썬의 floate 타입으로 변환.
            # 훈련 과정에서 각 배치를 처리할 때마다 이 줄이 실행되어, 각 배치의 손실 값을 train_loss 리스트에 순차적으로 추가
                    
        # 각 에포크마다 validation함수를 호출하여서 검증 세트에서 모델의 성능을 평가
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss) # 각 배치에서 계산된 모든 손실 값의 평균을 구함
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val F1 Score : [{_val_score:.5f}]')

        # scheduler이 설정되어 있다면 검증 성능에 따라 학습률을 조정
        if scheduler is not None:
            scheduler.step(_val_score)
            
        # 가장 좋은 성능을 보인 모델을 반환
        if best_score < _val_score:
            best_score = _val_score
            best_model = model
    
    return best_model

def validation(model, criterion, val_loader, device):
    model.eval() # 평가모드
    val_loss = []
    preds, true_labels = [], []

    # 평가모드의 경우에는 gradient를 초기화하는 부분이 없음 (backward 필요없음. 오직 평가만!)
    with torch.no_grad(): # 이 블록 내에서 그레디언트 계산을 중단하여, 필요하지 않은 메모리 사용을 줄이고 계산 속도 향상.
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.long().to(device)  # 데이터 타입 long으로 변경한 후 device로 올림 (int로 변경하였을 때, error 발생했음)
            
            pred = model(imgs)
            
            loss = criterion(pred, labels)
            
            # pred는 모델이 반환한 예측값. 각 클래스에 대한 확률 또는 점수를 포함하는 텐서. argmax(1)은 각 샘플에 대해 가장 높은 점수를 가진 클래스의 인덱스를 찾아줌.
            # detach()는 현재 계산 그래프로부터 이 텐서를 분리하여, 이후 연산이 그래프에 기록되지 않도록함. 메모리 사용량 줄임
            # cpu()는 cpu로 옮김 (GPU에 있었다면)
            # numpy()는 텐서를 numpy 배열로 변환
            # tolist()는 numpy 배열을 파이썬 리스트로 변환
            preds += pred.argmax(1).detach().cpu().numpy().tolist()

            # 실제 라벨도 위와 동일한 과정 진행
            true_labels += labels.detach().cpu().numpy().tolist()
            
            val_loss.append(loss.item())
        
        _val_loss = np.mean(val_loss)
        # average = 'macro'는 F1점수를 계산할 때, 각 클래스에 대한 F1점수를 동일한 가중치로 평균내어 전체 클래스에 대한 평균 F1점수를 계산.
        # 각 클래스의 샘플 크기와 관계없이 모든 클래스를 동등하게 취급. 이는 클래스 불균형이 있을 때 유용하며, 모든 클래스를 공평하게 평가하고자 할 때 사용.
        _val_score = f1_score(true_labels, preds, average='macro')
    
    return _val_loss, _val_score


model = Net() # 모델은 basemodel 가져옴
model.eval() #평가모드로 전환 (훈련모드가 아닌 평가모드를 불러온 이유가 뭐지?..)
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"]) # optimizer 'adam'으로 설정 / 학습률 위의 하이퍼파라미터

#학습률을 동적으로 조정하는 스케줄러 설정. 검증 성능이 개선되지 않으면 학습률 감소.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)



test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad(): # gradient 초기화 없이 평가 진행
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    
    preds = le.inverse_transform(preds)
    return preds

preds = inference(infer_model, test_loader, device)


submit = pd.read_csv('./sample_submission.csv')
submit['label'] = preds
submit.to_csv('./baseline_submit.csv', index=False)
