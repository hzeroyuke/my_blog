

一致性模型的Loss计算

```python
        # 两个相邻时间步
        t1 = torch.rand(128) * 0.9 + 0.1
        t2 = torch.clamp(t1 - 0.1, min=0.01)
        
        # 加噪声
        noise = torch.randn_like(batch)
        x1 = batch + t1.view(-1, 1) * noise
        x2 = batch + t2.view(-1, 1) * noise
        
        # 一致性损失：两个预测应该接近
        loss = F.mse_loss(model(x1, t1), model(x2, t2))
```

一致性模型的训练内容如下，取两个时间步，并且让两个时间步预测出来的内容尽可能相似

```python
class MinimalConsistencyModel(nn.Module):
    """最小一致性模型 - 只包含核心逻辑"""
    
    def __init__(self, dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 64), nn.ReLU(),
            nn.Linear(64, dim)
        )
    
    def forward(self, x, t):
        """x: 数据, t: 时间步"""
        # 拼接输入
        inp = torch.cat([x, t.view(-1, 1)], dim=1)
        
        # Skip connection: 当t→0时输出→x
        return (1 - t.view(-1, 1)) * x + t.view(-1, 1) * self.net(inp)
```

并且确保在t=0的时候，输出为真实图片即可

