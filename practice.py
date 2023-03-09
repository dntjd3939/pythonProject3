# 필요한 도구들 import
import torch    #파이토치 임포트
import torch.nn as nn   #파이토치의 신경망 모듈을 포함하는 패키지
import torchvision.datasets as dsets    #파이토치 토치비전 데이터셋에 mnist 가져올라고
import torchvision.transforms as transforms #텐서에 초기값을 주기위해
import torch.nn.init

# cpu gpu 선택
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 랜덤 시드 고정(다음에 돌렸을 때도 같은 결과 값을 얻기위해) 777은 매우 일반적인 값이다
torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

# parameters 값을 이렇게 정한 이유는???
# 기울기의 이동 STEP이고 너무 크면 오버슈팅 작으면 학습속도가 느려짐 대부분 0.01로 시작한 후 조정한다.
learning_rate = 0.001
# EPOCH는 전체 데이터셋에 대해 역전파 알고리즘 진행이 완료되는 횟수
training_epochs = 15
# EPOCH를 돌 때 메모리의 한계와 속도 저하를 막기 위해 전체 데이터를 쪼개서 학습하는데 그 쪼개진 데이터의 크기를 BATCH SIZE라 한다
batch_size = 100

# MNIST dataset 데이터를 가져오고
mnist_train = dsets.MNIST(root='MNIST_data/',  #다운로드 경로지정
                          train = True,  # True를 지정하면 훈련 데이터로 다운로드
                          transform = transforms.ToTensor(),  #텐서로 변환
                          download = True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                          train = False,   # False로 지정하면 테스트 데이터로 다운로드
                          transform = transforms.ToTensor(),
                          download = True)

# data loder 불러온 데이터셋을 크기를 정해서 가져온다
data_loader = torch.utils.data.DataLoader(dataset = mnist_train,
                                          batch_size = batch_size,
                                          shuffle = True, # 순서를 섞는다
                                          drop_last=True) # 남는 batch를 사용하지 않는다

# cnn 모델
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 첫번째층
        # ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  #크기를 반으로 줄인다?
        )
        # 두번째층
        # ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 전결합층 7x7x64 inputs -> 10 outputs  두번째 풀링층에서 나온 출력값을 입력으로 받아 10개의 클래스로 분류
        self.fc = nn.Linear(7 * 7 * 64, 10, bias=True)
        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):   #순전파 정의
        out = self.layer1(x) #x가 layer1 통과
        out = self.layer2(out) #통과한 out이 layer2 통과
        out = out.view(out.size(0), -1)  # 전 결합층을 위해서 배치 사이즈로 flatten view 함수를 사용하여 out의 크기를 배치 사이즈와 채널 수를 제외한 부분을 모두 하나의 차원으로 펼칩
        out = self.fc(out)  #전 결합층 통과
        return out


model = CNN().to(device)  #정해진 device로 모델 생성

#loss func 손실함수의 최솟값을 찾는것
criterion = nn.CrossEntropyLoss().to(device) #크로스엔트로피 로스 펑션 사용
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 아담 옵티마이저 사용

# training
total_batch = len(data_loader) # 전체의 배치 수 확인 가능
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader: # 미니 배치 단위로 꺼내온다. X는 미니 배치(인풋 데이터), Y는 레이블(라벨데이터). PyTorch에서 제공하는 데이터 로더를 사용하여 데이터셋을 불러올 때, 데이터셋에서 가져온 각 배치 데이터를 X와 Y로 분리하여 사용하는 코드입니다.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        # gpu사용할때 사용
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()   #모델의 파라미터에 대한 그래디언트 값을 0으로 초기화하여 다음 배치에서 올바르게 그래디언트 값을 계산할 수 있도록 해주는 함수
        hypothesis = model(X)   #학습된 모델을 사용하여 입력 데이터에 대한 예측값을 계산

        cost = criterion(hypothesis, Y)  # 모델의 예측값과 실제값 간의 손실(loss)을 계산, criterion은 손실 함수(loss function)를 의미하며, hypothesis는 모델의 예측값이고, Y는 실제값
        cost.backward()  # 손실 함수를 미분하여 모델의 파라미터에 대한 기울기(gradient)를 계산하는 코드
        optimizer.step()  # 백워드 값에 맞춰 옵티마이저를 스텝한다. backward() 메서드를 호출하여 구한 기울기 값을 이용하여 최적화 함수가 구현한 알고리즘에 따라 모델의 파라미터를 업데이트


        avg_cost += cost / total_batch  #에버리지 코스트 계산해서 쌓기

    print('[Epoch:{}] cost = {}'.format(epoch+1, avg_cost))  #에폭 코스트 확인
print('Learning Finished!')


# test
with torch.no_grad():  #학습을 안하니깐 노 그래디언트 선언
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)   ##데이터를 가져오고 view 함수를 사용하여 데이터를 (데이터 개수, 채널 수, 높이, 너비) 형태로 변환하며, float()을 통해 데이터 타입을 float으로 변환하고, to(device)를 통해 GPU를 사용할 수 있는 디바이스로 데이터를 옮겨줍니다.
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)  #모델에다가 x데이터를 한번에 다집어 넣어 프개딕션을 계산한다
    correct_prediction = torch.argmax(prediction, 1) == Y_test  # 프래딕션에서 가장 큰값을 가진 인덱스를 y test와 비교하여 올바르게 분류된 예측값의 개수를 구한다
    accuracy = correct_prediction.float().mean()  #이값을 전체 데이터셋 크기로 나눠 평균을 낸다
    print('Accuracy:', accuracy.item())

