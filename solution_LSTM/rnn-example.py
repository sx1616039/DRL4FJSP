import torch
import torch.nn as nn


# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 定义输入数据和标签
input_size = 10
hidden_size = 32
num_layers = 2
output_size = 2
batch_size = 3
seq_lengths = [5, 3, 7]

# 生成随机输入数据
input_data = [torch.randn(seq_lengths[i], input_size) for i in range(batch_size)]

# 对输入数据进行填充，使其长度统一
padded_input_data = torch.nn.utils.rnn.pad_sequence(input_data, batch_first=True)

# 创建模型实例
model = RNN(input_size, hidden_size, num_layers, output_size)

# 将模型移至GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 将输入数据和标签移至GPU（如果可用）
padded_input_data = padded_input_data.to(device)

# 前向传播
output = model(padded_input_data)

print(output)