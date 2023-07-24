'''
# https://github.com/rammyram/image_captioning/tree/master
임베딩 : 고차원의 데이터를 저차원의 실수 벡터로 표현하는 기법으로, 모델에 비구조적인 데이터를 입력으로 제공하고 처리하는 데 사용
hidden size : LSTM 모델의 숨겨진 상태(hidden state)의 크기. 하이퍼 파라미터.
'''

import torch
torch.cuda.empty_cache()
print(torch.cuda.is_available())

"""---------------------vocabulary 클래스-----------------------------"""
import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter

# Vocabulary 클래스 정의
class Vocabulary(object):
    def __init__(self,
                 vocab_threshold,  # 단어 최소 등장 빈도
                 vocab_file='./vocab.pkl',
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 annotations_file='',
                 vocab_from_file=False):  # 기존 단어 집합 파일 사용 여부

        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    # 파일에서 어휘를 로드하거나 처음부터 어휘를 빌드하는 함수
    def get_vocab(self):
        # 파일이 존재하고 vocab_from_file이 True인 경우 파일 열고 저장된 어휘 로드
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:  # vocab_file을 바이너리 모드로 열고 저장된 어휘를 피클 형식으로 로드.
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx  # 단어 -> 인덱스
                self.idx2word = vocab.idx2word  # 인덱스 -> 단어
            print('Vocabulary successfully loaded from vocab.pkl file!')
        # 파일이 없으면 어휘 구축 후 vocab파일에 피클 형식으로 저장
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:  # 바이너리 모드로 열고 피클형식으로 저장.
                pickle.dump(self, f)

    # 토큰을 정수로 변환하는 사전을 채우는 함수
    def build_vocab(self):
        self.init_vocab()  # 어휘 초기화.(다음에 바로 있음)
        self.add_word(self.start_word)  # 추가(다음에 바로 있음)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()  # 캡션 데이터에서 단어를 추출하여 어휘를 추가. 캡션 데이터에서 등장하는 모든 단어를 어휘에 등록(다음에 있음)

    # 토큰을 정수로 변환하는 사전을 초기화하는 함수
    def init_vocab(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    # 어휘에 토큰을 추가하는 함수
    def add_word(self, word):
        if len(word) > 0 and not word in self.word2idx:  # 토큰의 길이가 0보다 크고, 어휘에 등록되지 않은 경우에만 처리
            self.word2idx[word] = self.idx  # 딕셔너리 [단어, 인덱스값]
            self.idx2word[self.idx] = word  # 딕셔너리 [인덱스, 단어]
            self.idx += 1  # 인덱스에 1을 더해 다음 토큰에 대한 인덱스 준비.

    # 훈련 캡션을 반복하고 임계값 이상인 모든 토큰을 어휘에 추가하는 함수
    def add_captions(self):
        coco = COCO(self.annotations_file)
        counter = Counter()  # 빈도수 체크를 위한 counter
        ids = coco.anns.keys()

        for i, id in enumerate(ids):  # id목록을 반복하면서
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if
                 cnt >= self.vocab_threshold and len(word) > 0]  # 등장 빈도가 임계값 이상인 토큰들을 선택합니다
        # 선택된 토큰들 어휘에 추가
        for i, word in enumerate(words):
            self.add_word(word)

    # 주어진 단어에 대한 호출 가능한 함수
    def __call__(self, word):
        # 단어가 어휘에 없다면 unk로 반환, 있으면 단어 반환
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    # 어휘 단어 개수 반환
    def __len__(self):
        return len(self.word2idx)

"""---------------------데이터로더 및 데이터셋-----------------------------"""
import nltk
import os
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

# 데이터로더 정의
def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0):

    assert mode in ['train', 'val', 'test'], "mode must be one of 'train' 'val' 'test'."
    if vocab_from_file == False:
        assert mode == 'train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # mode가 train일 경우
    if mode == 'train':
        # vocab_from_file이 존재한다면
        if vocab_from_file == True:
            assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join('D:/gwangho_pj1_data/coco2017/train2017')                           # 이미지 폴더 경로
        annotations_file = os.path.join('D:/gwangho_pj1_data/coco2017/annotations/captions_train2017.json')  # 어노테이션1 경로

    if mode == 'val':
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join('D:/gwangho_pj1_data/coco2017/val2017')                           # 이미지 폴더 경로
        annotations_file = os.path.join('D:/gwangho_pj1_data/coco2017/annotations/captions_val2017.json')  # 어노테이션1 경로

    if mode == 'test':   # test 모드일 경우
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file == True, "Change vocab_from_file to True."
        img_folder = os.path.join('D:/gwangho_pj1_data/coco2017/test2017')                      # 이미지 경로
        annotations_file = os.path.join('D:/gwangho_pj1_data/coco2017/annotations/image_info_test2017') # 어노테이션 경로

    # COCODataset 불러서 dataset생성
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)
    # train모드이면
    if mode == 'train' or 'val':
        # 랜덤하게 캡션 길이를 선택하고 해당 길이의 인덱스를 샘플링합니다.
        indices = dataset.get_train_indices()
        # 샘플링된 인덱스로 배치를 가져올 배치 샘플러를 생성하고 할당합니다.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset,
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader


class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word,
                 end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder

        if self.mode == 'train' or 'val':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens if len(token) > 0]  # 0 길이의 캡션 제외
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]

    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train' or 'val':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Check if image file exists
            if not os.path.exists(os.path.join(self.img_folder, path)):
                print("Image file does not exist:", path)

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        # obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train'or'val':
            return len(self.ids)
        else:
            return len(self.paths)

"""---------------------확인하는 부분(딱히 필요없음)-----------------------------"""
import sys
from pycocotools.coco import COCO
import nltk
nltk.download('punkt')
from torchvision import transforms
# (Optional) TODO #2: Amend the image transform below.
transform_train = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Set the minimum word count threshold.
vocab_threshold = 6
# Specify the batch size.
batch_size = 64
# Obtain the data loader.
train_data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)

val_data_loader = get_loader(transform=transform_train,
                         mode='val',
                         batch_size=1)

train_dataset = train_data_loader.dataset
val_dataset = val_data_loader.dataset

print("Total number of train images:", len(train_dataset))
print("Total number of val images:", len(val_dataset))


from collections import Counter
import matplotlib.pyplot as plt

# 캡션 길이별 총 캡션 개수를 집계
counter = Counter(train_data_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))

# Extract caption lengths and their frequencies
x = [pair[0] for pair in lengths]  # 캡션 길이
y = [pair[1] for pair in lengths]  # 캡션 개수

# Plot the caption length distribution
plt.bar(x, y)
plt.xlabel('train Caption Length')
plt.ylabel('Number of Captions')
plt.title('Caption Length Distribution')
plt.show()

# 캡션 길이별 총 캡션 개수를 집계
counter = Counter(val_data_loader.dataset.caption_lengths)
lengths = sorted(counter.items(), key=lambda pair: pair[1], reverse=True)
for value, count in lengths:
    print('value: %2d --- count: %5d' % (value, count))

# Extract caption lengths and their frequencies
x = [pair[0] for pair in lengths]  # 캡션 길이
y = [pair[1] for pair in lengths]  # 캡션 개수

# Plot the caption length distribution
plt.bar(x, y)
plt.xlabel('train Caption Length')
plt.ylabel('Number of Captions')
plt.title('Caption Length Distribution')
plt.show()


# import numpy as np
# import torch.utils.data as data
#
# # 훈련 데이터에서 랜덤하게 인덱스(이미지/캡션)를 샘플링하여 가져옵니다.
# indices = data_loader.dataset.get_train_indices()
# print('sampled indices:', indices)
#
# # 새로운 배치 샘플러를 생성하고 데이터로더의 배치 샘플러에 새로운 샘플러로 할당
# new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
# data_loader.batch_sampler.sampler = new_sampler
#
# # Obtain the batch.
# images, captions = next(iter(data_loader))
#
# print('images.shape:', images.shape)
# print('captions.shape:', captions.shape)




"""---------------------모델-----------------------------"""
import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''
이미지를 인코딩하기 위한 CNN기반의 인코더입니다. 주어진 이미지를 입력으로 받아 해당 이미지의 특징을 추출하고, 그 특징을 임베딩된 벡터로 변환하여 반환합니다.

이 모델은 사전 훈련된 ResNet-50 아키텍처를 사용합니다. 모델의 초기화 단계에서 사전 훈련된 ResNet-50을 로드하고, 그 중 일부 계층의 파라미터를 동결시킵니다.
이로써 사전 훈련된 가중치를 고정시키고, 이후 추가로 학습되는 파라미터들은 모델이 이미지 특징을 잘 추출하는 데 도움을 줄 수 있습니다.

모듈 초기화 후에는 ResNet의 마지막 계층을 제외한 모든 계층을 사용하여 이미지의 특징을 추출하는데 사용됩니다. 
이 추출된 특징은 1차원 벡터로 펼쳐진 후, 선형 변환을 통해 임베딩된 벡터로 매핑됩니다.
이때, 선형 변환 이후에는 배치 정규화(Batch Normalization) 계층이 적용되어 임베딩된 벡터를 정규화합니다.

이 모델은 입력 이미지를 받아 처리한 후, 임베딩된 이미지 특징 벡터를 반환합니다.
'''
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)  # 사전 훈련된 resnet50 사용
        for param in resnet.parameters():          # resnet 파라미터 동결 -> 사전 훈련 가중치 고정
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]    # resnet의 마지막 계층을 제외한 모든 계층을 list(moduels)에 저장
        self.resnet = nn.Sequential(*modules)     # 마지막 계층을 제외한 모든 계층을 nn.sequential객체 생성.
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)  # resnet의 마지막계층(fully connected layer)의 입력 특징 개수와 embed_size를 가진 선형변환모듈 생성. -> 이미지 특징을 임베딩된 벡터로 매핑하는 역활
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)   # 배치정규화 수행.
        self.embed.weight.data.normal_(0., 0.02)    # 가중치와 편향 초기화
        self.embed.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)           # 이미지를 받아 resnet에 전달하여 특징 추출(마지막 계층 제외한 pretrained된 resnet50)
        features = features.view(features.size(0), -1)   # features를 2d 텐서로 변환(4d->2d). 미니베치 차원을 유지, 나머지는 펼쳐서 2d로 만든다. features.size(0)이 미니배치(batch_size) 크기.
        features = self.embed(features)          # 변환된 특징을 embed 선형변환에 적용. -> 임베딩된 벡터로 매핑
        return features


'''
이미지 특성과 캡션을 입력으로 받아, 캡션의 다음 단어를 예측하는 역할을 수행합니다. 모델은 LSTM(Long Short-Term Memory) 네트워크를 기반으로 구성되어 있습니다.
'''
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.hidden_size = hidden_size      # 히든사이즈 변수 선언(하이퍼 파리미터)
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)   # 임베딩 레이어. 단어를 지정된 크기의 벡터로 변환하는 역활.
        # lstm 모델
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=1,            # LSTM레이어의 개수.
                            bias=True,               # 입력과 숨겨진 상태 사이의 편향 사용여부
                            batch_first=True,        # 입력과 출력의 첫번째 차원을 배치 크기로 설정할것인가
                            dropout=0,               # 드롭아웃 비율
                            bidirectional=False,     # 양방향 LSTM사용여부
                            )
        self.linear = nn.Linear(hidden_size, vocab_size)   # 선형 레이어. 숨겨진 상태의 출력 차원을 출력할 단어 수로 매핑.

    def init_hidden(self, batch_size):
        # LSTM레이어의 초기 은닉상태를 생성하는 메소드. LSTM 레이어의 초기 은닉 상태는 이전 시점의 정보가 없는 첫 번째 시점에서 시작하기 위해 사용됩니다.
        # LSTM 레이어의 초기 은닉 상태를 담은 튜플(h_0, c_0)을 리턴.
        # h_0: 형태 (1, batch_size, hidden_size)의 0으로 채워진 텐서
        # c_0: 형태 (1, batch_size, hidden_size)의 0으로 채워진 텐서
        return (torch.zeros((1, batch_size, self.hidden_size), device=device),
                torch.zeros((1, batch_size, self.hidden_size), device=device))

    def forward(self, features, captions):
        captions = captions[:, :-1]   # lstm에 입력될 때 예측을 피하기 위해 <end>를 제거

        self.batch_size = features.shape[0]     # features형태(batch_size, embed_size)
        self.hidden = self.init_hidden(self.batch_size)  # 은닉상태 초기화

        embeddings = self.word_embeddings(captions) # 캡션의 각 단어에 대해 임베딩된 단어 벡터를 생성. embeddings의 새로운 형태 : (batch_size, captions 길이 - 1, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # 특징과 캡션을 쌓는다. embeddings의 새로운 형태 : (batch_size, caption 길이, embed_size)

        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # 임베딩과 은닉상태를 입력으로 받아 출력과 은닉상태를 얻는다. # lstm_out의 형태 : (batch_size, caption 길이, hidden_size)

        outputs = self.linear(lstm_out)  # Fully connected layer. outputs의 형태 : (batch_size, caption 길이, vocab_size)

        return outputs

    ## Greedy search. 사전 처리된 이미지 텐서(inputs)를 입력으로 받아 예측된 문장(길이가 max_len인 텐서 ID의 리스트)을 반환합니다.
    def sample(self, inputs):
        output = []
        batch_size = inputs.shape[0]  # 배치 크기는 추론 시 1입니다. inputs의 형태 : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size)  # LSTM의 초기 은닉 상태를 얻습니다.

        while True:  # end 나오기 전까지 lstm 돌리면서 예측 단어 인덱스 저장.
            lstm_out, hidden = self.lstm(inputs, hidden)  # lstm_out의 형태 : (1, 1, hidden_size)
            outputs = self.linear(lstm_out)   # outputs의 형태 : (1, 1, vocab_size)
            outputs = outputs.squeeze(1)      # outputs의 형태 : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1)  # max_index : 예측된 단어의 인덱스

            output.append(max_indice.cpu().numpy()[0].item())  # NumPy 배열로 변환한 후, [0]을 통해 첫 번째 요소를 가져옵니다. 그리고 .item()을 사용하여 해당 요소를 스칼라 값으로 변환.
                                                               # 스칼라 값은 예측된 단어의 인덱스를 나타내며, output 리스트에 추가

            if (max_indice == 1):          # max_indice == 0 : <start>, max_indice == 1 : <end>. 따라서 end를 예측 시 break.
                break
            # 마지막 예측된 단어를 새로운 입력으로 LSTM에 임베딩하기 위해 준비합니다.
            inputs = self.word_embeddings(max_indice)  # inputs의 형태 : (1, embed_size)
            inputs = inputs.unsqueeze(1)  # inputs의 형태 : (1, 1, embed_size)

        return output

"""---------------------train/val-----------------------------"""
import torch
import torch.nn as nn
from torchvision import transforms
import sys
import math

batch_size = 64  # batch size
vocab_threshold = 6  # minimum word count threshold
vocab_from_file = True  # if True, load existing vocab file
embed_size = 512  # dimensionality of image and word embeddings
hidden_size = 512  # number of features in hidden state of the RNN decoder
num_epochs = 100  # number of training epochs (1 for testing)
save_every = 1  # determines frequency of saving model weights
print_every = 200  # determines window for printing average loss
log_file = 'D:/gwangho_pj1_data/training_log.txt'  # name of file with saved training loss and perplexity

vocab_size = len(train_data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

criterion = nn.CrossEntropyLoss().to(device)
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters())
optimizer = torch.optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08)
total_step = math.ceil(len(train_data_loader.dataset.caption_lengths) / batch_size)


import torch.utils.data as data
import numpy as np
import os
import requests
import time
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# Open the training log file.
f = open(log_file, 'w')

losses = []
perplexities = []
val_losses = []
val_perplexities = []
epoch_losses = []
epoch_perplexities = []
bleu_1_scores = []
bleu_2_scores = []
bleu_3_scores = []
bleu_4_scores = []
best_train_loss = float('inf')
best_val_loss = float('inf')
last_epoch = num_epochs

for epoch in range(1, num_epochs + 1):
    epoch_loss = 0.0
    epoch_perplexity = 0.0

    references = []  # Initialize references for each epoch
    hypotheses = []  # Initialize hypotheses for each epoch


    for i_step in range(1, total_step + 1):
        # Randomly sample a caption length, and sample indices with that length.
        train_indices = train_data_loader.dataset.get_train_indices()
        val_indices = val_data_loader.dataset.get_train_indices()

        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        train_sampler = data.sampler.SubsetRandomSampler(indices=train_indices)
        val_sampler = data.sampler.SubsetRandomSampler(indices=val_indices)

        train_data_loader.batch_sampler.sampler = train_sampler
        val_data_loader.batch_sampler.sampler = val_sampler

        # Obtain the batch for training.
        train_images, train_captions = next(iter(train_data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        train_images = train_images.to(device)
        train_captions = train_captions.to(device)

        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()

        # Pass the inputs through the CNN-RNN model.
        features = encoder(train_images)
        outputs = decoder(features, train_captions)

        # Calculate the training batch loss.
        train_loss = criterion(outputs.contiguous().view(-1, vocab_size), train_captions.view(-1))

        # Backward pass.
        train_loss.backward()

        # Update the parameters in the optimizer.
        optimizer.step()

        stats = 'Epoch [%d/%d], Step [%d/%d], Train Loss: %.4f, Train Perplexity: %5.4f' % (
        epoch, num_epochs, i_step, total_step, train_loss.item(), np.exp(train_loss.item()))

        # Append training loss and perplexity values.
        losses.append(train_loss.item())
        perplexities.append(np.exp(train_loss.item()))

        epoch_loss += train_loss.item()
        epoch_perplexity += np.exp(train_loss.item())

        # Print training statistics for every print_every steps
        if i_step % print_every == 0:
            print(stats)

    # Calculate average training loss and perplexity per epoch.
    epoch_loss /= total_step
    epoch_perplexity /= total_step

    epoch_losses.append(epoch_loss)
    epoch_perplexities.append(epoch_perplexity)

    with torch.no_grad():
        val_loss = 0.0
        val_perplexity = 0.0

        for i_step, (val_images, val_captions) in enumerate(val_data_loader, 1):
            # Move batch of images and captions to GPU if CUDA is available.
            val_images = val_images.to(device)
            val_captions = val_captions.to(device)

            # Pass the inputs through the CNN-RNN model.
            val_features = encoder(val_images)
            val_outputs = decoder(val_features, val_captions)

            # Calculate the validation batch loss.
            val_loss += criterion(val_outputs.contiguous().view(-1, vocab_size), val_captions.view(-1)).item()

            # Convert indices to words for each sampled caption
            sampled_captions = [
                [val_data_loader.dataset.vocab.idx2word[idx] for idx in caption] for caption in val_outputs.argmax(dim=-1).cpu().numpy()
            ]

            # Convert references to words
            references.extend(
                [
                    [val_data_loader.dataset.vocab.idx2word[idx] for idx in caption] for caption in val_captions.tolist()
                ]
            )

            # Add the first generated caption as hypothesis
            hypotheses.extend(sampled_captions)


            print("\nPredicted caption: {}".format(sampled_captions[0]))
            print("Actual caption: {}".format([val_data_loader.dataset.vocab.idx2word[idx] for idx in val_captions[0].tolist()]))

        # Calculate average validation loss and perplexity per epoch.
        val_loss /= len(val_data_loader)
        val_perplexity = np.exp(val_loss)

        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)

        # Calculate BLEU scores
        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)


        # Append BLEU scores
        bleu_1_scores.append(bleu_1)
        bleu_2_scores.append(bleu_2)
        bleu_3_scores.append(bleu_3)
        bleu_4_scores.append(bleu_4)


        # Print BLEU scores
        print("BLEU-1: {:.4f}".format(bleu_1))
        print("BLEU-2: {:.4f}".format(bleu_2))
        print("BLEU-3: {:.4f}".format(bleu_3))
        print("BLEU-4: {:.4f}".format(bleu_4))


        # Get validation statistics.
        val_stats = 'Epoch [%d/%d], Val Loss: %.4f, Val Perplexity: %5.4f' % (
        epoch, num_epochs, val_loss, val_perplexity)

        # Print validation statistics.
        print('\n' + val_stats)

    if train_loss.item() < best_train_loss:
        best_train_loss = train_loss.item()
        torch.save(decoder.state_dict(), os.path.join('D:/gwangho_pj1_data/coco2017/coco_only_weight', 'best_train_loss_decoder.pkl'))
        torch.save(encoder.state_dict(), os.path.join('D:/gwangho_pj1_data/encdec', 'best_train_loss_encoder.pkl'))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(decoder.state_dict(), os.path.join('D:/gwangho_pj1_data/coco2017/coco_only_weight', 'best_val_loss_decoder.pkl'))
        torch.save(encoder.state_dict(), os.path.join('D:/gwangho_pj1_data/coco2017/coco_only_weight', 'best_val_loss_encoder.pkl'))

    if epoch == num_epochs:
        last_epoch = epoch
        torch.save(decoder.state_dict(), os.path.join('D:/gwangho_pj1_data/coco2017/coco_only_weight', 'last_epoch_decoder.pkl'))
        torch.save(encoder.state_dict(), os.path.join('D:/gwangho_pj1_data/coco2017/coco_only_weight', 'last_epoch_encoder.pkl'))

# Close the training log file.
f.close()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

# 가장 좋은 훈련 손실과 검증 손실의 에폭과 손실 값을 찾습니다
best_train_loss_epoch = epoch_losses.index(min(epoch_losses)) + 1
best_train_loss_value = min(epoch_losses)
best_val_loss_epoch = val_losses.index(min(val_losses)) + 1
best_val_loss_value = min(val_losses)

# 가장 좋은 훈련 손실과 검증 손실의 위치를 강조합니다
plt.scatter(best_train_loss_epoch, best_train_loss_value, c='r', marker='o', label=f'Best Train Loss: {best_train_loss_value:.4f}')
plt.scatter(best_val_loss_epoch, best_val_loss_value, c='g', marker='o', label=f'Best Val Loss: {best_val_loss_value:.4f}')

plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), epoch_perplexities, label='Train Perplexity')
plt.plot(range(1, num_epochs + 1), val_perplexities, label='Val Perplexity')
plt.xlabel('Epoch')
plt.ylabel('Perplexity')
plt.title('Training and Validation Perplexity')
plt.legend()
plt.show()

# Plotting BLEU scores
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), bleu_1_scores, label='BLEU-1')
plt.plot(range(1, num_epochs + 1), bleu_2_scores, label='BLEU-2')
plt.plot(range(1, num_epochs + 1), bleu_3_scores, label='BLEU-3')
plt.plot(range(1, num_epochs + 1), bleu_4_scores, label='BLEU-4')
plt.xlabel('Epochs')
plt.ylabel('BLEU Score')
plt.title('BLEU Scores over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# TODO #1: Define a transform to pre-process the testing images.
transform_test = transforms.Compose([transforms.Resize((224, 224)), \
                                     transforms.ToTensor(), \
                                     transforms.Normalize((0.485, 0.456, 0.406), \
                                                          (0.229, 0.224, 0.225))])

# -#-#-# Do NOT modify the code below this line. #-#-#-#

# Create the data loader.
data_loader = get_loader(transform=transform_test,
                         mode='test')

import numpy as np
import matplotlib.pyplot as plt

# Obtain sample image before and after pre-processing.
orig_image, image = next(iter(data_loader))

# Visualize sample image, before pre-processing.
plt.imshow(np.squeeze(orig_image))
plt.title('example image')
plt.show()

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO #2: Specify the saved models to load.
encoder_file = 'D:/gwangho_pj1_data/coco2017/coco_only_weight/best_val_loss_encoder.pkl'
decoder_file = 'D:/gwangho_pj1_data/coco2017/coco_only_weight/best_val_loss_decoder.pkl'

# TODO #3: Select appropriate values for the Python variables below.
embed_size = 512
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('./encdec', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./encdec', decoder_file)))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)

# Move image Pytorch Tensor to GPU if CUDA is available.
image = image.to(device)

# Obtain the embedded image features.
print("image.shape: ", image.shape)
features = encoder(image).unsqueeze(1)
print("features.shape: ", features.shape)
print()

# Pass the embedded image features through the model to get a predicted caption.
output = decoder.sample(features)
print('example output:', output)

assert (type(output) == list), "Output needs to be a Python list"
assert all([type(x) == int for x in output]), "Output should be a list of integers."
assert all([x in data_loader.dataset.vocab.idx2word for x in
            output]), "Each entry in the output needs to correspond to an integer that indicat"


# TODO #4: Complete the function.
def clean_sentence(output):
    list_string = []

    for idx in output:
        list_string.append(data_loader.dataset.vocab.idx2word[idx])

    list_string = list_string[1:-1]  # Discard <start> and <end> words
    sentence = ' '.join(list_string)  # Convert list of string to full string
    sentence = sentence.capitalize()  # Capitalize the first letter of the first word
    return sentence


sentence = clean_sentence(output)
print('example sentence:\n', sentence)

assert type(sentence) == str, 'Sentence needs to be a Python string!'


import matplotlib.pyplot as plt

def get_prediction():
    num_rows = 5
    num_cols = 6
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 12))

    for i, ax in enumerate(axes.flat):
        orig_image, image = next(iter(data_loader))
        ax.imshow(np.squeeze(orig_image))
        ax.axis('off')

        image = image.to(device)
        features = encoder(image).unsqueeze(1)
        output = decoder.sample(features)
        sentence = clean_sentence(output)
        ax.set_title(sentence)

    plt.tight_layout()
    plt.show()


get_prediction()
