# coding:utf-8
import torch.optim
from pylab import *
import os
import dataset_L2 as dataset
import network_L3 as network
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 忽略级别2及以下消息（级别1是提示，级别2是警告，级别3是错误）
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'  # ''0,1,2,3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 进入工程路径

# 参数设置
SIZE = 224
BATCH_SIZE = 64
NUM_CLASS = 2
is_sampling = 'no_sampler'  # # 训练集采样模式：over_sampler   down_sampler   no_sampler
is_train = False  # True-加载训练模型，False-加载测试模型
is_pretrained = True  # 是否加载预训练权重

# 路径
model_path = os.path.join('model')
PATH = "data/labels.csv"
TEST_PATH = ''
result_path = "model/ensemble_result.csv"

# 模型列表
model_list = [
    'alexnet',
    # 'resnet18',
    'vgg16',
    'densenet',
    # 'inception'
]

# 模型路径名
name_list = [
    'my_model/alexnet/L3_alexnet_best_model.pkl',
    # 'my_model/resnet18/L3_resnet18_best_model.pkl',
    'my_model/vgg16/L3_vgg16_best_model.pkl',
    'my_model/densenet/L3_densenet_best_model.pkl'
]

# 画图名字
model_plot_list = copy.deepcopy(model_list)


def for_test_resnet(m, n):
    print('--------------')
    print(n)
    model = network.initialize_model(m, pretrained=is_pretrained)
    name = n
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(name), False)
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
            import torch.nn.functional as F
            prob = F.softmax(output.data, dim=1)  # softmax[[0.9,0.1],[0.8,0.2]]
            test_prob = np.append(test_prob, prob.detach().cpu().numpy(), axis=0)
            test_pred = np.hstack((test_pred, predicted.detach().cpu().numpy()))
            test_true = np.hstack((test_true, labels.detach().cpu().numpy()))

    images = test_loader.dataset.test_img
    test_acc = 100 * metrics.accuracy_score(test_true, test_pred)
    test_AUC = metrics.roc_auc_score(y_true=test_true, y_score=test_pred)  # 按标签计算auc

    print('Accuracy of the network is: %.4f %%' % test_acc)
    print('Test_AUC: %.4f' % test_AUC)
    test_classification_report = metrics.classification_report(test_true, test_pred, digits=4)
    # tn, fp, fn, tp = metrics.confusion_matrix(test_true, test_pred).ravel()
    print('test_classification_report\n', test_classification_report)
    # print('TN=%d, FP=%d, FN=%d, TP=%d' % (tn, fp, fn, tp))

    return test_acc, images, test_true, test_pred, test_prob, test_AUC


def show_multi_roc(model_path, model_plot_list, true_list, prob_list, auc_list):
    # 画平均ROC曲线的两个参数
    mean_tpr = 0.0  # 用来记录画平均ROC曲线的信息
    mean_fpr = np.linspace(0, 1, 100)
    cnt = 0
    for model_name, true, probas_, auc_ in zip(model_plot_list, true_list, prob_list, auc_list):
        cnt += 1
        fpr, tpr, thresholds = roc_curve(true, probas_[:, 1])  # 该函数得到伪正例、真正例、阈值，这里只使用前两个
        mean_tpr += np.interp(mean_fpr, fpr, tpr)  # 插值函数 interp(x坐标,每次x增加距离,y坐标)  累计每次循环的总值后面求平均值
        mean_tpr[0] = 0.0  # 将第一个真正例=0 以0为起点
        roc_auc = auc(fpr, tpr)  # 求auc面积

        plt.plot(fpr, tpr, lw=1, label='{0} (area = {1:.3f})'.format(model_name, auc_))  # 画出当前分割数据的ROC曲线

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Reference')  # 画对角线

    mean_tpr /= cnt  # 求数组的平均值
    mean_tpr[-1] = 1.0  # 坐标最后一个点为（1,1）  以1为终点
    # mean_auc = auc(mean_fpr, mean_tpr)  # 按概率计算auc
    mean_auc = np.mean(auc_list)  # 按标签计算auc
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = {0:.3f})'.format(mean_auc), lw=1)

    plt.xlim([0.00, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
    plt.ylim([0.00, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC curves on test set')
    plt.legend(loc="lower right")
    plt.savefig('{}/{}.jpg'.format(model_path, 'ROC-AUC curves on test set'))
    plt.show()


models_pred = []
models_true = []
models_prob = []
models_auc = []  # 按标签计算auc

# 加载数据
train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE, is_train=is_train,
                                                            is_sampling=is_sampling)
# 测试集加载模型
for i in range(len(model_list)):
    if model_list[i] == 'inception':
        SIZE = 299
        print(model_list[i], SIZE)
        train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE,
                                                                    is_train=is_train, is_sampling=is_sampling)
        dataset.get_mean_std(test_loader)

    # 测试集
    test_acc, test_img, test_true, test_pred, test_prob, test_AUC = for_test_resnet(model_list[i], name_list[i])
    models_pred.append(test_pred)
    models_true.append(test_true)
    models_prob.append(test_prob)
    models_auc.append(test_AUC)

    if model_list[i] == 'inception' and model_list[-1] != 'inception':  # 加载完Inception后，恢复SIZE=224
        SIZE = 224
        train_loader, val_loader, test_loader = dataset.get_dataset(PATH, TEST_PATH, SIZE, BATCH_SIZE,
                                                                    is_train=is_train,
                                                                    is_sampling=is_sampling)
# 绘制多模型ROC曲线
show_multi_roc(model_path, model_plot_list, models_true, models_prob, models_auc)

# 集成投票
print('-->Ensemble Learning!')
ceil = math.ceil(len(models_pred) / 2)  # 上取整
y_predict = np.array(np.sum(models_pred, axis=0) >= ceil, dtype='int')  # axis=0按列求和
vote_acc = 100 * metrics.accuracy_score(test_true, y_predict)
test_classification_report = metrics.classification_report(test_true, y_predict, digits=4)
vote_auc = metrics.roc_auc_score(y_true=test_true, y_score=y_predict)
tn, fp, fn, tp = metrics.confusion_matrix(test_true, y_predict).ravel()
print(test_classification_report)
print('model_list:', model_list)
print('Vote_acc = %.4f %%' % vote_acc)
print('Vote_AUC = %.4f' % vote_auc)
print('Vote_tn, fp, fn, tp = ', (tn, fp, fn, tp))
# print('y_predict', y_predict)

# 输出测试结果图片
show_batch = 64
iters = math.ceil(test_img.shape[0] / show_batch)
begin = 0
for iter in range(iters):
    end = begin + show_batch if (begin + show_batch) <= test_img.shape[0] else test_img.shape[0]
    show_test_img, show_test_true, show_test_pred = test_img[begin:end], test_true[begin:end], y_predict[begin:end]
    dataset.show_test(show_test_img, show_test_true, show_test_pred, show_batch, iter)
    begin = end
    end = begin + show_batch
    plt.savefig('{}/{}.jpg'.format(model_path, (str(iter + 1)).rjust(4, '0')))
