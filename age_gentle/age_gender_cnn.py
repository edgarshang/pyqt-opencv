import torch
from torch.utils.data import DataLoader

# # 检查是否可以利用GPU
# train_on_gpu = torch.cuda.is_available()
# if not train_on_gpu:
#     print('CUDA is not available.')
# else:
#     print('CUDA is available!')


class MyMulitpleTaskNet(torch.nn.Module):
    def __init__(self):
        super(MyMulitpleTaskNet, self).__init__()
        self.cnn_layers = torch.nn.Sequential(
            # 卷积层 (64x64x3的图像)
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2, 2),

            # 32x32x32
            torch.nn.Conv2d(64, 96, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(96),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(96, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2, 2),

            # 64x64x16
            torch.nn.Conv2d(128, 196, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(196),
            torch.nn.MaxPool2d(2, 2)
        )
        # 全局最大池化
        self.global_max_pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        # linear layer (8*8 -> 10)

        self.age_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(196, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 1),
            torch.nn.Sigmoid()
        )

        self.gender_fc_layers = torch.nn.Sequential(
            torch.nn.Linear(196, 25),
            torch.nn.ReLU(),
            torch.nn.Linear(25, 2)
        )

    def forward(self, x):
        # stack convolution layers
        x = self.cnn_layers(x)

        # 2x2x196
        B, C, H, W = x.size()
        out = self.global_max_pooling(x).view(B, -1)

        # 全连接层
        out_age = self.age_fc_layers(out)
        out_gender = self.gender_fc_layers(out)
        return out_age, out_gender


# if __name__ == "__main__":
#     # create a complete CNN
#     model = MyMulitpleTaskNet()
#     print(model)

#     # 使用GPU
#     if train_on_gpu:
#         model.cuda()

#     ds = AgeGenderDataset("D:/python/pytorch_tutorial/UTKFace/")
#     num_train_samples = ds.num_of_samples()
#     bs = 16
#     dataloader = DataLoader(ds, batch_size=bs, shuffle=True)

#     # 训练模型的次数
#     num_epochs = 25
#     # optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
#     model.train()

#     # 损失函数
#     mse_loss = torch.nn.MSELoss()
#     cross_loss = torch.nn.CrossEntropyLoss()
#     index = 0
#     for epoch in  range(num_epochs):
#         train_loss = 0.0
#         for i_batch, sample_batched in enumerate(dataloader):
#             images_batch, age_batch, gender_batch = \
#                 sample_batched['image'], sample_batched['age'], sample_batched['gender']
#             if train_on_gpu:
#                 images_batch, age_batch, gender_batch = images_batch.cuda(), age_batch.cuda(), gender_batch.cuda()
#             optimizer.zero_grad()

#             # forward pass: compute predicted outputs by passing inputs to the model
#             m_age_out_, m_gender_out_ = model(images_batch)
#             age_batch = age_batch.view(-1, 1)
#             gender_batch = gender_batch.long()

#             # calculate the batch loss
#             loss = mse_loss(m_age_out_, age_batch) + cross_loss(m_gender_out_, gender_batch)

#             # backward pass: compute gradient of the loss with respect to model parameters
#             loss.backward()

#             # perform a single optimization step (parameter update)
#             optimizer.step()

#             # update training loss
#             train_loss += loss.item()
#             if index % 100 == 0:
#                 print('step: {} \tTraining Loss: {:.6f} '.format(index, loss.item()))
#             index += 1

#             # 计算平均损失
#         train_loss = train_loss / num_train_samples

#         # 显示训练集与验证集的损失函数
#         print('Epoch: {} \tTraining Loss: {:.6f} '.format(epoch, train_loss))

#     # save model
#     model.eval()
#     torch.save(model, 'age_gender_model.pt')
