import torch
import torch.nn as nn
import torch.optim as optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # 对第一个输入进行特征提取
        self.dense1 = nn.Linear(7, 64)
        
        # 对第二个输入进行特征提取
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3))
        
        # 计算conv1输出的维度
        # 假设输入是 (N, 1, 10, 25)
        # 输出是 (N, 32, 8, 23)
        self.flattened_dim = 32 * 8 * 23

        # 第一个输出头：命令选择（10类命令）
        self.command_output = nn.Linear(64 + self.flattened_dim, 10)
        
        # 第二个输出头：第一个数字（可选，可能为三位数字）
        self.num1_output = nn.Linear(64 + self.flattened_dim, 3)
        
        # 第三个输出头：第二个数字（可选）
        self.num2_output = nn.Linear(64 + self.flattened_dim, 3)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input1, input2):
        # 对第一个输入进行特征提取
        x1 = self.relu(self.dense1(input1))
        
        # 对第二个输入进行特征提取
        # input2需要增加一个channel维度
        x2 = input2.unsqueeze(1) # (N, 1, 10, 25)
        x2 = self.relu(self.conv1(x2))
        x2 = x2.view(x2.size(0), -1) # Flatten
        
        # 合并两个输入的特征
        combined = torch.cat((x1, x2), dim=1)
        
        # 输出头
        command = self.softmax(self.command_output(combined))
        num1 = self.relu(self.num1_output(combined))
        num2 = self.relu(self.num2_output(combined))
        
        return command, num1, num2

# 创建模型实例
model = MyModel()

# 定义损失函数和优化器
criterion_command = nn.CrossEntropyLoss()
criterion_num = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 示例输入
input1_sample = torch.randn(1, 7)
input2_sample = torch.randn(1, 10, 25)

# 前向传播
command_pred, num1_pred, num2_pred = model(input1_sample, input2_sample)

# 假设的真实标签
command_true = torch.tensor([1], dtype=torch.long)
num1_true = torch.tensor([[1., 2., 3.]])
num2_true = torch.tensor([[4., 5., 6.]])

# 计算损失
loss_command = criterion_command(command_pred, command_true)
loss_num1 = criterion_num(num1_pred, num1_true)
loss_num2 = criterion_num(num2_pred, num2_true)
total_loss = loss_command + loss_num1 + loss_num2

# 反向传播和优化
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

print("Model, loss, and optimizer are set up using PyTorch.")