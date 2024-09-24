
1. 运行 train_L3.py 训练模型脚本，训练完成后模型保存到“CaseStudy/model”文件夹下
"""
    修改train_L3.py训练模型的参数，分别设置：
        backbone = 'alexnet'
        backbone = 'resnet18'
        backbone = 'vgg16'
    执行步骤1三次，训练三次模型，在./model/文件夹下得到
        model/alexnet/L3_alexnet_best_model.pkl
        model/resnet18/L3_resnet18_best_model.pkl
        model/vgg16/L3_vgg16_best_model.pkl
    最后，运行ensemble_L4.py
"""
2. 将 train_L3.py 该脚本中 is_train 改为False，可以测试模型
3. 运行 ensenble_L4.py，投票法集成多个模型