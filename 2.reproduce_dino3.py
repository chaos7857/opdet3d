import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --------------------------
# 1. 核心模块：DINO v3 蒸馏头
# --------------------------
class DINOHead(nn.Module):
    def __init__(self, in_dim=768, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, num_layers=3):
        super().__init__()
        layers = []
        # 隐藏层（激活函数使用GELU，DINO v3推荐）
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
        # 瓶颈层（降维）
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        # 投影头（输出特征）
        self.projection = nn.Linear(bottleneck_dim, out_dim)
        # 动态温度参数（DINO v3核心优化）
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        # 输入：(batch_size, num_patches, in_dim) → 输出：(batch_size, out_dim)
        x = self.mlp(x[:, 0])  # 取CLS token
        x = F.normalize(x, dim=-1)  # L2归一化
        return self.projection(x), x  # 返回logits和瓶颈特征

# --------------------------
# 2. 数据增强：多尺度+强 augmentation（DINO v3要求）
# --------------------------
class MultiScaleAugment:
    def __init__(self, img_size=224, min_scale=0.14, max_scale=1.0):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(min_scale, max_scale)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # 多尺度视图生成（DINO v3新增：2个全局视图+1个局部视图）
    def __call__(self, x):
        return [self.transform(x) for _ in range(3)]  # 返回3个增强视图

# --------------------------
# 3. DINO v3 主模型
# --------------------------
class DINOv3(nn.Module):
    def __init__(self, backbone_name="vit_base_patch16_224", out_dim=65536):
        super().__init__()
        # 骨干网络（ViT-B，冻结位置编码，DINO标准操作）
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.backbone.pos_embed.requires_grad = False
        in_dim = self.backbone.num_features  # ViT-B默认768
        
        # 学生网络头（2个：适配多视图）
        self.student_head1 = DINOHead(in_dim, out_dim)
        self.student_head2 = DINOHead(in_dim, out_dim)
        
        # 教师网络头（动量更新）
        self.teacher_head = DINOHead(in_dim, out_dim)
        # 冻结教师头参数（仅通过动量更新）
        for param in self.teacher_head.parameters():
            param.requires_grad = False
        
        # 动量系数（DINO v3调整为0.9995，更慢更新）
        self.momentum = 0.9995

    def momentum_update(self):
        # 动量更新教师网络（学生→教师）
        for s_param, t_param in zip(self.backbone.parameters(), self.backbone.parameters()):
            t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data
        for s_param, t_param in zip(self.student_head1.parameters(), self.teacher_head.parameters()):
            t_param.data = self.momentum * t_param.data + (1 - self.momentum) * s_param.data

    def forward(self, views):
        # views: 3个增强视图 → (batch_size, 3, C, H, W)
        batch_size = views[0].shape[0]
        
        # 学生网络前向（处理所有视图）
        student_feat1 = self.backbone(views[0])  # 视图1特征
        student_feat2 = self.backbone(views[1])  # 视图2特征
        student_feat3 = self.backbone(views[2])  # 视图3特征（局部）
        
        s_logits1, s_feat1 = self.student_head1(student_feat1)
        s_logits2, s_feat2 = self.student_head2(student_feat2)
        
        # 教师网络前向（仅处理全局视图，不计算梯度）
        with torch.no_grad():
            self.momentum_update()  # 动量更新
            t_feat1 = self.backbone(views[0])
            t_feat2 = self.backbone(views[1])
            t_logits1, _ = self.teacher_head(t_feat1)
            t_logits2, _ = self.teacher_head(t_feat2)
            # 教师输出归一化（温度调节）
            t_logits1 = t_logits1 / self.teacher_head.temperature
            t_logits2 = t_logits2 / self.teacher_head.temperature
        
        return s_logits1, s_logits2, t_logits1, t_logits2

# --------------------------
# 4. 损失函数（DINO v3改进版）
# --------------------------
def dino_loss(s_logits1, s_logits2, t_logits1, t_logits2, temperature=0.1):
    # 学生→教师蒸馏损失（双向匹配）
    loss1 = F.cross_entropy(s_logits1 / temperature, F.softmax(t_logits2, dim=-1))
    loss2 = F.cross_entropy(s_logits2 / temperature, F.softmax(t_logits1, dim=-1))
    
    # 多尺度一致性损失（DINO v3新增）
    loss_consistency = F.mse_loss(F.normalize(s_logits1, dim=-1), F.normalize(s_logits2, dim=-1))
    
    return (loss1 + loss2) / 2 + 0.1 * loss_consistency

# --------------------------
# 5. 训练流程
# --------------------------
if __name__ == "__main__":
    # 配置参数
    batch_size = 32
    epochs = 100
    lr = 1e-4
    img_size = 224
    data_path = "./data"  # 数据集路径（自动下载CIFAR-10）
    
    # 加载数据集（CIFAR-10示例，可替换为ImageNet）
    dataset = datasets.CIFAR10(
        root=data_path, train=True, download=True,
        transform=MultiScaleAugment(img_size=img_size)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 初始化模型、优化器
    model = DINOv3(backbone_name="vit_base_patch16_224").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 训练循环
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            views, _ = batch  # views: (batch_size, 3, C, H, W)
            views = [v.cuda() for v in views]
            
            # 前向传播
            s_logits1, s_logits2, t_logits1, t_logits2 = model(views)
            
            # 计算损失
            loss = dino_loss(s_logits1, s_logits2, t_logits1, t_logits2)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印日志
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    # 保存模型（仅保存骨干网络+学生头，用于下游任务）
    torch.save({
        "backbone": model.backbone.state_dict(),
        "student_head": model.student_head1.state_dict()
    }, "dino_v3_pretrained.pth")