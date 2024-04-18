import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        
        # 定义卷积层
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # 定义全连接层
        self.fc1 = nn.Linear(in_features=6144, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        # 卷积操作
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        # 展平
        x = torch.flatten(x, start_dim=1)
        # 全连接层
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        # x = F.softmax(x, dim=1)
        return x
 
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        # 添加dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        
        # 获取batch大小
        batch_size = x.size(0)
        q = q.view(batch_size, 2, self.num_heads // 2, self.head_size)
        k = k.view(batch_size, 2, self.num_heads // 2, self.head_size)
        v = v.view(batch_size, 2, self.num_heads // 2, self.head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # 使用torch.div()进行缩放
        scores = torch.div(scores, torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)))
        scores = nn.Softmax(dim=-1)(scores)
        
        # 应用dropout
        scores = self.dropout(scores)
        x = torch.matmul(scores, v)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc(x)
        return x
 
class MultiHeadClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads, hidden_size):
        super(MultiHeadClassifier, self).__init__()
        self.num_heads = num_heads
        self.d_model = input_dim
        self.attention = MultiHeadAttention(input_dim, num_heads)
        self.fc = nn.Linear(input_dim, hidden_size)
        self.BN = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.BN(x)
        x1 = x
        x = self.attention(x)
        x = x.mean(dim=1)
        x += x1
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        # x = F.softmax(x, dim=1)
        return x