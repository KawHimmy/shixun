import torch.optim
from pylab import *
import os
import dataset_L2 as dataset

from sklearn import metrics
import torch.nn.functional as F
import network_L3 as network
import math
from sklearn.metrics import precision_recall_curve, classification_report, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score

import time

start111 = time.time()

"""
    修改train_L3.py训练模型的参数，分别设置：
        backbone = 'alexnet'
        backbone = 'resnet18'
        backbone = 'vgg16'
    训练三次模型，在./model/文件夹下得到
        model/alexnet/L3_alexnet_best_model.pkl
        model/resnet18/L3_resnet18_best_model.pkl
        model/vgg16/L3_vgg16_best_model.pkl
    最后，运行ensemble_L4.py
"""

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5'
# device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
start = datetime.datetime.now()
print('Device:', device)

# 参数设置
is_sampling = 'no_sampler'  # 训练集采样模式： over_sampler-上采样  down_sampler-下采样  no_sampler-无采样
is_train = False # True-训练模型  False-测试模型
is_pretrained = True  # 是否加载预训练权重
backbone = 'vgg16'  # 骨干网络：alexnet resnet18 vgg16 densenet inception
model_path = 'my_model/' + backbone  # 模型存储路径

# 训练参数设置
SIZE = 299 if backbone == 'inception' else 224  # 图像进入网络的大小
BATCH_SIZE = 1 # batch_size数
NUM_CLASS = 2  # 分类数
EPOCHS = 100  # 迭代次数

# 进入工程路径并新建文件夹
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 进入工程路径
dataset.mkdir('model')  # 新建文件夹

if is_train:  # 训练模式
    PATH = 'data/final_files.csv'
    TEST_PATH = ''
else:  # 测试模型
    PATH = 'data/labels.csv'
    TEST_PATH = ''
    best_model_name = os.path.join(model_path, 'L3_%s_best_model.pkl' % backbone)
    print('best_model_name=', best_model_name)

# 加载数据
dataset.mkdir(model_path)
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=is_train,
                                                            is_sampling=is_sampling)

model = network.initialize_model(backbone, pretrained=is_pretrained, NUM_CLASS=NUM_CLASS)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=6e-3)  # 更新所有层权重
# criterion = torch.nn.CrossEntropyLoss()
class_weights = torch.tensor([0.3, 0.7]).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

train_batch = 3


# train_resnet
def train_resnet(model):
    history_train = []
    history_valid = []
    history_auc = []
    best_auc = 0.
    best_epoch = 0
    no_improvement_epochs = 0
    patience = 1

    # 动态lr设置
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(EPOCHS):
        correct = total = 0.
        loss_list = []
        # 为教学使用，仅选择部分数据进行训练，通过train_batch参数控制
        for batch_index, (batch_x, batch_y) in enumerate(train_loader, 0):
            # if batch_index < train_batch: #  修改了选择全部数据进行训练
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            model.train()
            # 优化过程
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

            # 输出训练结果
            loss_list.append(loss.item())
            _, predicted = torch.max(output.data, 1)  # 返回每行的最大值
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        train_avg_acc = 100 * correct / total
        train_avg_loss = np.mean(loss_list)
        print('[Epoch=%d/%d]Train set: Avg_loss=%.4f, Avg_accuracy=%.4f%%' % (
            epoch + 1, EPOCHS, train_avg_loss, train_avg_acc))
        history_train.append((train_avg_loss, train_avg_acc))

        # change the learning rate by scheduler
        scheduler.step()

        # 验证集
        valid_pred, valid_true, auc, valid_acc, valid_avg_loss = valid_resnet(model)
        print('[Epoch=%d/%d]Validation set: Avg_loss=%.4f, Avg_accuracy=%.4f%%, AUC=%.4f' %
              (epoch + 1, EPOCHS, valid_avg_loss, valid_acc, auc))
        history_valid.append((valid_avg_loss, valid_acc))
        history_auc.append(auc)

        if auc > best_auc:
            best_epoch = epoch
            no_improvement_epochs = 0
            # 保存最优模型
            best_model_name = os.path.join(model_path, 'L3_%s_best_model.pkl' % backbone)
            print('>>>>>>>>>>>>>>Best model is %s' % (str(best_epoch + 1) + '.pkl'))
            torch.save(model.state_dict(), best_model_name)  # 训练多GPU，测试多GPU
            # torch.save(model.module.state_dict(), best_model_name)  # 训练多GPU，测试单GPU
            best_auc = auc
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print('>>>>>>>>>>>>>>No improvement for %d epochs. Training stopped.' % patience)
            break

    print("Train finished!")
    print('Train running time = %s' % str(datetime.datetime.now() - start))
    print('Saving last model...')
    last_model_name = os.path.join(model_path, 'L3_%s_last_model.pkl' % backbone)
    torch.save(model.state_dict(), last_model_name)  # 训练多GPU，测试多GPU

    return best_model_name, history_train, history_valid, history_auc


def valid_resnet(model):
    # print('------ Validation Start -----')
    with torch.no_grad():
        model.eval()
        val_loss_list = []
        valid_pred = []
        valid_true = []
        valid_prob = np.empty(shape=[0, 2])  # 概率值

        for batch_index, (batch_valid_x, batch_valid_y) in enumerate(val_loader, 0):
            if torch.cuda.is_available():
                batch_valid_x, batch_valid_y = batch_valid_x.cuda(), batch_valid_y.cuda()
            output = model(batch_valid_x)
            _, batch_valid_pred = torch.max(output.data, 1)
            prob = F.softmax(output.data, dim=1)  # prob=softmax[[0.9,0.1],[0.8,0.2]]
            loss = criterion(output, batch_valid_y)
            val_loss_list.append(loss.item())
            valid_pred = np.hstack((valid_pred, batch_valid_pred.detach().cpu().numpy()))
            valid_true = np.hstack((valid_true, batch_valid_y.detach().cpu().numpy()))
            valid_prob = np.append(valid_prob, prob.detach().cpu().numpy(), axis=0)  # valid_prob=概率列表=[N*2]

        valid_avg_loss = np.mean(val_loss_list)
        valid_acc = 100 * metrics.accuracy_score(valid_true, valid_pred)
        valid_AUC = metrics.roc_auc_score(y_true=valid_true, y_score=valid_prob[:, 1])  # y_score=正例的概率=[N*1]
        # valid_AUC = metrics.roc_auc_score(y_true=valid_true, y_score=valid_pred)
        tn, fp, fn, tp = metrics.confusion_matrix(valid_true, valid_pred).ravel()
        valid_classification_report = metrics.classification_report(valid_true, valid_pred, digits=4)
    return valid_pred, valid_true, valid_AUC, valid_acc, valid_avg_loss


# Precision-Recall 曲线
import pandas as pd
import datetime
def plot_precision_recall_curve(true_labels, predicted_probs, model_path, save_data=False):
    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs[:, 1])

    # 保存数据
    if save_data:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pr_data = pd.DataFrame({'Recall': recall, 'Precision': precision})
        pr_data.to_csv(f'{model_path}/pr_curve_data_{timestamp}.csv', index=False)

    plt.figure()
    plt.plot(recall, precision, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.savefig(f'{model_path}/precision_recall_curve.jpg')
    plt.show()


# 柱状图：Precision、Recall、F1 Score
def plot_classification_report(true_labels, predicted_labels, model_path):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)

    metrics_dict = {'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure()
    plt.bar(names, values, color=['blue', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title('Classification Metrics')
    plt.savefig(f'{model_path}/classification_metrics.jpg')
    plt.show()


# 概率直方图
def plot_probability_histogram(predicted_probs, model_path):
    plt.figure()
    plt.hist(predicted_probs[:, 1], bins=50, color='blue', alpha=0.7, label='Predicted Probability for Class 1')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Distribution of Predictions')
    plt.savefig(f'{model_path}/probability_histogram.jpg')
    plt.show()


def for_test_resnet(best_model_name):
    print('------ Testing Start ------')
    model.load_state_dict(torch.load(best_model_name), False)
    test_pred = []
    test_true = []
    test_prob = np.empty(shape=[0, 2])  # 概率值

    with torch.no_grad():
        model.eval()
        for test_x, test_y in test_loader:
            if torch.cuda.is_available():
                images, labels = test_x.cuda(), test_y.cuda()
            else:
                images, labels = test_x, test_y
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            prob = F.softmax(output.data, dim=1)  # softmax[[0.9,0.1],[0.8,0.2]]
            test_prob = np.append(test_prob, prob.detach().cpu().numpy(), axis=0)
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    images = test_loader.dataset.test_img
    test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
    test_AUC = metrics.roc_auc_score(y_true=test_true, y_score=test_prob[:, 1])  # y_score=正例的概率
    # test_AUC = metrics.roc_auc_score(y_true=test_true, y_score=test_pred)
    test_classification_report = metrics.classification_report(test_true, test_pred, digits=4)
    tn, fp, fn, tp = metrics.confusion_matrix(test_true, test_pred).ravel()
    print('test_classification_report\n', test_classification_report)
    print('Accuracy of the network is: %.4f %%' % test_acc)
    print('Test_AUC: %.4f' % test_AUC)
    print('TN=%d, FP=%d, FN=%d, TP=%d' % (tn, fp, fn, tp))

    # 可视化 Precision-Recall 曲线
    plot_precision_recall_curve(test_true, test_prob, model_path, save_data=True)

    # 绘制柱状图：Precision、Recall、F1 Score
    plot_classification_report(test_true, test_pred, model_path)

    # 绘制概率直方图
    # plot_probability_histogram(test_prob, model_path)

    return test_acc, images, test_true, test_pred, test_prob


if is_train:
    # 训练集
    best_model_name, history_train, history_valid, history_auc = train_resnet(model)
    # 绘制训练集和验证集的loss、acc、AUC
    dataset.show_plot(history_train, history_valid, history_auc, model_path)

#测试集
test_acc, test_img, test_true, test_pred , test_prob= for_test_resnet(best_model_name)
show_batch = 100
iters = math.ceil(test_img.shape[0] / show_batch)
begin = 0
for iter in range(iters):
    end = begin + show_batch if (begin + show_batch) <= test_img.shape[0] else test_img.shape[0]
    show_test_img, show_test_true, show_test_pred = test_img[begin:end], test_true[begin:end], test_pred[begin:end]
    dataset.show_test(show_test_img, show_test_true, show_test_pred, show_batch, iter)
    begin = end
    end = begin + show_batch
    plt.savefig('{}/{}.jpg'.format(model_path, (str(iter + 1).rjust(4, '0'))))

print('头部ct运行时间level3alexnet：', time.time()-start111)

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def plot_confusion_matrix(true_labels, predicted_labels, classes, model_path):
    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, predicted_labels)

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    # 保存并显示混淆矩阵图
    plt.savefig(f'{model_path}/confusion_matrix.jpg')
    plt.show()


# 获得 test_true 和 test_pred
test_acc, test_img, test_true, test_pred , test_prob= for_test_resnet(best_model_name)

# 定义类标签
classes = ['True', 'bad']

# 绘制并保存混淆矩阵
plot_confusion_matrix(test_true, test_pred, classes, model_path)
print(1111)

