# 필요한 도구들 import
import torch    #pytorch import
import torch.nn as nn   #pytorch 신경망 모듈을 포함하는 패키지
import torchvision.datasets as dsets    #pytorch dataset import
import torchvision.transforms as transforms     #다양한 형태로 변환 가능
import torch.nn.init    #tensor 초기값 입력
from torch.optim import lr_scheduler    #scheduler import

# cpu gpu 선택
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 랜덤 시드 고정(항상 같은 결과 값을 얻기 위해)
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


# 기울기의 이동 STEP
learning_rate = 0.001
# 전체 데이터셋에 대해 알고리즘 진행이 완료되는 횟수
training_epochs = 15
# 쪼개진 데이터의 크기
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',  #다운로드 경로지정
                          train = True,  # True로 지정하여 훈련 데이터로 다운로드
                          transform = transforms.ToTensor(),  #텐서로 변환
                          download = True)

mnist_test = dsets.MNIST(root='MNIST_data/',  #다운로드 경로지정
                          train = False,   # False로 지정하여 테스트 데이터로 다운로드
                          transform = transforms.ToTensor(),  #텐서로 변환
                          download = True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset = mnist_train,    #dataset 훈련데이터 사용
                                          batch_size = batch_size,  #batch_size 지정
                                          shuffle = True,   # 순서를 섞는다
                                          drop_last=True)   # 계산 후 남는 batch를 사용하지 않음

# 신경망 클래스 정의
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   #합성곱(in_channel = 1, out_channel = 32, kernel_size=3, stride=1, padding=1)
            nn.ReLU(),      #ReLU 활성화 함수 사용
            nn.MaxPool2d(2)  #맥스풀링(kernel_size=2, stride=2)
        )
        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  #합성곱(in_channel = 32, out_channel = 64, kernel_size=3, stride=1, padding=1)
            nn.ReLU(),      #ReLU 활성화 함수 사용
            nn.MaxPool2d(2)  #맥스풀링(kernel_size=2, stride=2)
        )
        # 세번째층
        # ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  #합성곱(in_channel = 64, out_channel = 128, kernel_size=3, stride=1, padding=1)
            nn.ReLU(),      #ReLU 활성화 함수 사용
            nn.MaxPool2d(2)  #맥스풀링(kernel_size=2, stride=2)
        )
        # 전결합층 4x4x128 inputs -> 10 outputs
        self.fc = nn.Linear(4 * 4 * 128, 10, bias=True)
        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):   #순전파 정의
        out = self.layer1(x) #x가 layer1 통과
        out = self.layer2(out) #통과한 out이 layer2 통과
        out = self.layer3(out) #통과한 out이 layer3 통과
        out = out.view(out.size(0), -1)  # 전결합층을 위해 flatten
        out = self.fc(out)  #전결합층 통과
        return out

#CNN 모델 정의
model = CNN().to(device)

#LOSS FUNC 정의
criterion = nn.CrossEntropyLoss().to(device)
#optimizer 정의
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#StepLR scheduler 정의
scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.4)

# training
total_batch = len(data_loader)      # 전체의 배치 수 확인
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:    # X는 미니 배치(인풋 데이터), Y는 레이블(라벨데이터)
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()   #모델의 파라미터에 대한 그래디언트 값을 0으로 초기화
        hypothesis = model(X)   #입력 데이터에 대한 예측값 계산

        cost = criterion(hypothesis, Y)  # 모델의 예측값과 실제값 간의 손실(loss)을 계산
        cost.backward()  # 모델의 파라미터에 대한 gradient를 계산
        optimizer.step()  # 기울기 값을 이용하여 최적화 함수가 구현한 알고리즘에 따라 모델의 파라미터를 업데이트

        avg_cost += cost / total_batch  #average cost 계산

    scheduler.step()    #step_size 마다 학습률을 조정함.

    print('[Epoch:{}], cost = {}, Learning Rate: {}'.format(epoch+1, avg_cost, scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
print('Learning Finished!')


# test
with torch.no_grad():  # no gradient 선언
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)   #데이터를 (데이터 개수, 채널 수, 높이, 너비) 형태 및 float 타입으로 변환
    Y_test = mnist_test.test_labels.to(device)  #정답 label을 DEVICE로 옮김

    prediction = model(X_test)  #모델에다가 x데이터를 한번에 넣어 prediction 계산
    correct_prediction = torch.argmax(prediction, 1) == Y_test  # prediction에서 가장 큰값을 가진 인덱스를 y test와 비교하여 올바르게 분류된 예측값의 개수를 구함
    accuracy = correct_prediction.float().mean()  #전체 데이터셋 크기로 나눠 평균값을 구함
    print('Accuracy:', accuracy.item())

