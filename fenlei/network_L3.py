import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoConfig


class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def initialize_model(backbone, pretrained, NUM_CLASS=2):
    if backbone == "resnet18":
        # 加载 Hugging Face 的预训练 ResNet18 模型
        config = AutoConfig.from_pretrained('pretrained_resnet18')
        model = AutoModel.from_pretrained('pretrained_resnet18', config=config)

        # ResNet18 输出的特征维度为 512
        num_ftrs = config.hidden_sizes[-1] if hasattr(config, 'hidden_sizes') else 512

        # 定义自定义的分类头
        classification_head = ClassificationHead(num_ftrs, NUM_CLASS)

        # 定义一个新的模型，将预训练模型和自定义分类头组合在一起
        class CustomResNet(nn.Module):
            def __init__(self, backbone, classification_head):
                super(CustomResNet, self).__init__()
                self.backbone = backbone
                self.classifier = classification_head

            def forward(self, x):
                # 获取从 backbone（ResNet18）提取的特征
                outputs = self.backbone(x)
                # 从输出中提取池化后的特征
                pooled_output = outputs.last_hidden_state  # Hugging Face 的模型输出
                # 将提取的特征传入分类头
                return self.classifier(pooled_output)

        # 返回组合后的模型
        model_conv = CustomResNet(model, classification_head)

    elif backbone == "alexnet":
        model_conv = models.alexnet(pretrained=pretrained)
        num_ftrs = model_conv.classifier[6].in_features
        model_conv.classifier[6] = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone.startswith("vgg"):
        model_conv = models.__dict__[f'vgg{backbone.split("vgg")[-1]}'](pretrained=pretrained)
        num_ftrs = model_conv.classifier[6].in_features
        model_conv.classifier[6] = nn.Linear(num_ftrs, NUM_CLASS)

    elif backbone == "densenet":
        model_conv = models.densenet121(pretrained=pretrained)
        num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = ClassificationHead(num_ftrs, NUM_CLASS)

    elif backbone == "inception":
        model_conv = models.inception_v3(pretrained=pretrained)
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = ClassificationHead(num_ftrs, NUM_CLASS)
        model_conv.aux_logits = False  # 禁用辅助输出

    else:
        raise ValueError(
            f"Unsupported backbone: {backbone}. Please choose from resnet18, alexnet, vgg, densenet, inception.")

        # If available, move model to GPU
    model = model_conv.cuda() if torch.cuda.is_available() else model_conv
    return model
