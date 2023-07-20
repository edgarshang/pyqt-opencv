import torch
import torch.nn.functional as F
# from custom_dataset import FaceLandmarksDataset
from torch.utils.data import Dataset, DataLoader

# 检查是否可以利用GPU
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available!')


# 全局最大池化，但是还不是全局深度最大池化
class GlobalMaxPool2d(torch.nn.Module):
    def __init__(self):
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, x):
        assert len(x.size()) == 4, x.size()
        B, C, H, W = x.size()
        return F.max_pool2d(x, (W, H)).view(B, H*W)


class ChannelPool(torch.nn.MaxPool1d):
    def __init__(self, channels, isize):
        super(ChannelPool, self).__init__(channels)
        self.kernel_size = channels
        self.stride = isize

    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n,c,w*h).permute(0,2,1)
        pooled =  torch.nn.functional.max_pool1d(input, self.kernel_size, self.stride,
                        self.padding, self.dilation, self.ceil_mode,
                        self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0,2,1)
        return pooled.view(n,c,w,h).view(n, w*h)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            # 卷积层 (64x64x3的图像)
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(2, 2),
            # 32x32x32
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.Conv2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(2, 2),

            # 64x64x16
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            # torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.PReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.dw_max = ChannelPool(128, 8*8)
        # linear layer (16*16 -> 10)
        self.fc1 = torch.nn.Linear(128*8*8, 8*8)
        self.fc = torch.nn.Linear(8*8, 10)

    def forward(self, x):
        # stack convolution layers
        x = self.cnn_layers(x)
        # print("===========================================================x.shape", x.shape)
        n,c,h,w = x.shape
        # 16x16x128
        # 深度最大池化层
        out = self.dw_max(x)
        # x = x.view(n,c*h*w)
        # out = self.fc1(x)
        # print("===========================================================out.shape", out.shape)
        # out = x.view(-1, 128*64)
        # 全连接层
        out = self.fc(out)
        # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++out.shape", out.shape)
        return out


def myloss_fn(pred_y, target_y):
    target_y = target_y.view(-1, 10)
    sum = torch.zeros(len(target_y)).cuda()
    for i in range(0, len(target_y)):
        t_item = target_y[i]
        p_item = pred_y[i]
        dx = t_item[0] - t_item[2]
        dy = t_item[1] - t_item[3]
        id = torch.sqrt(dx*dx + dy*dy)
        # N个点求
        for t in range(0, len(t_item), 2):
            dx = p_item[t] - t_item[t]
            dy = p_item[t+1] - t_item[t+1]
            dist = torch.sqrt(dx*dx + dy*dy)
            sum[i] += (dist / id)
        sum[i] = sum[i] / 5
    return torch.sum(sum).cuda()


if __name__ == "__main__":
    # create a complete CNN
    model = Net()
    print(model)

    # 使用GPU
    if train_on_gpu:
        model.cuda()

    ds = FaceLandmarksDataset("./landmark.txt")
    num_train_samples = ds.num_of_samples()
    dataloader = DataLoader(ds, batch_size=4, shuffle=True)

    # 训练模型的次数
    num_epochs = 50
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        for i_batch, sample_batched in enumerate(dataloader):
            images_batch, landmarks_batch = \
                sample_batched['image'], sample_batched['landmarks']
            if train_on_gpu:
                images_batch, landmarks_batch = images_batch.cuda(), landmarks_batch.cuda()
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(images_batch)
            # calculate the batch loss
            loss = myloss_fn(output, landmarks_batch)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()
            # 计算平均损失
        train_loss = train_loss / num_train_samples

        # 显示训练集与验证集的损失函数
        print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

    # save model
    model.eval()
    torch.save(model, 'model_landmarks.pt')
