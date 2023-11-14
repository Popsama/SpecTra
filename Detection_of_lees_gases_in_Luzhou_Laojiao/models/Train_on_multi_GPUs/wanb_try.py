import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# 定义我们的模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 定义训练函数
def train():
    # 初始化wandb
    run = wandb.init()
    # 获取超参数
    config = run.config
    lr = config.lr
    epochs = config.epochs

    # 确定我们的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并移动到设备上
    model = Model().to(device)

    # 定义优化器和损失函数
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 随机生成一些训练数据，并移动到设备上
    x_train = torch.randn(100, 10).to(device)
    y_train = torch.randn(100, 1).to(device)

    # 开始训练
    for epoch in range(epochs):
        optimizer.zero_grad()  # 清空梯度
        outputs = model(x_train)  # 前向传播
        loss = loss_fn(outputs, y_train)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        # 在wandb中记录损失
        wandb.log({"Loss": loss.item()})

# 定义超参数搜索空间
sweep_config = {
    'method': 'grid',  # 可以是'grid', 'random'或'bayes'
    'metric': {
        'name': 'Loss',
        'goal': 'minimize'
    },
    'parameters': {
        'lr': {
            'values': [0.001, 0.01, 0.1]
        },
        'epochs': {
            'values': [50, 100, 150]
        }
    }
}

# 初始化超参数搜索
sweep_id = wandb.sweep(sweep_config, project="my-project")

# 执行超参数搜索
wandb.agent(sweep_id, function=train)