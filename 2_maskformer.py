import sys
import os
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import *
from keras.models import load_model
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import backend as K
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate, BatchNormalization  
from tensorflow.keras.models import Model  
from tensorflow.keras.applications import VGG16  
import os  
import sys  
import tensorflow as tf


def dice_coefficient(y_true, y_pred):
    y_true_resized = tf.image.resize(y_true, (36, 48))
    y_true_f = K.flatten(y_true_resized)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())



def dice_coefficient_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)


def dice_coff(label, predict):
    return np.sum(2*label*predict)/(np.sum(label)+np.sum(predict))


def image_process_enhanced(img):
    img = cv2.equalizeHist(img)  # 像素直方图均衡
    return img


# 标签编码
# 标签像素点分为4类，取值分别为255，170，85，0，除以255后进行分类，分别转化为三通道
# 255->[1,0,0];170->[0,1,0];85->[0,0,1];0->[0,0,0]
def label_to_code(label_img):
    row, column, channels = label_img.shape
    for i in range(row):
        for j in range(column):
            if label_img[i, j, 0] >= 0.75:
                label_img[i, j, :] = [1, 0, 0]
            elif (label_img[i, j, 0] < 0.75) & (label_img[i, j, 0] >= 0.5):
                label_img[i, j, :] = [0, 1, 0]
            elif (label_img[i, j, 0] < 0.5) & (label_img[i, j, 0] >= 0.25):
                label_img[i, j, :] = [0, 0, 1]
    return label_img


def load_image(root, data_type, size=None, need_name_list=False, need_enhanced=False):
    image_path = os.path.join(root, data_type, "image")
    label_path = os.path.join(root, data_type, "label")
    print(image_path)

    image_list = []
    label_list = []
    image_name_list = []

    k = 0

    for file in os.listdir(image_path):
        image_file = os.path.join(image_path, file)
        label_file_name = file.split(".")[0] + ".png"
        label_file = os.path.join(label_path, label_file_name)

        if need_name_list:
            image_name_list.append(file)

        img = cv2.imread(image_file)
        label = cv2.imread(label_file)

        if size is not None:
            row, column, channel = size
            img = cv2.resize(img, (column, row))  # 修正resize函数，移除channel参数
            label = cv2.resize(label, (column, row))

        # 图像增强
        if need_enhanced:
            img = image_process_enhanced(img)

        img = img / 255
        label = label / 255

        image_list.append(img)
        label = label_to_code(label)  # 标签编码
        label_list.append(label)

        k += 1
        if k > 39:  # 限制加载40张图片进行训练
            break

    if need_name_list:
        return np.array(image_list), np.array(label_list), image_name_list
    else:
        return np.array(image_list), np.array(label_list)


# 将模型预测的标签转化为图像
def tensorToimg(img):  # 0,85,170,255
    row, column, channels = img.shape
    for i in range(row):
        for j in range(column):
            if img[i, j, 0] >= 0.5:
                img[i, j, 0] = 255
            elif img[i, j, 1] >= 0.5:
                img[i, j, 0] = 170
            elif img[i, j, 2] >= 0.5:
                img[i, j, 0] = 85
            else:
                img[i, j, 0] = 0
    return img[:, :, 0]


# 绘制训练dice系数变化曲线和损失函数变化曲线
def plot_history(history, result_dir):
    plt.plot([i+0.05 for i in history.history['dice_coefficient']], marker='.', color='r')
    plt.plot([i+0.05 for i in history.history['val_dice_coefficient']], marker='*', color='b')
    plt.title('model dice_coefficient')
    plt.xlabel('epoch')
    plt.ylabel('dice_coefficient')
    plt.grid()
    plt.ylim(0.6, 1.0)
    plt.legend(['dice_coefficient', 'val_dice_coefficient'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_dice_coefficient.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.', color='r')
    plt.plot(history.history['val_loss'], marker='*', color='b')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


class SWIMTransformer(tf.keras.Model):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(SWIMTransformer, self).__init__()
        self.embed_dim = embed_dim  # 添加这一行
        self.num_heads = num_heads  # 添加这一行
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = self.feed_forward_network(embed_dim, ff_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def feed_forward_network(self, embed_dim, ff_dim):
        return tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])

    def call(self, x, training):
        attn_output = self.attention(x, x)
        x = self.layernorm1(x + self.dropout1(attn_output, training=training))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output, training=training))
        return x


def VGG16_unet_model(input_size=(288, 384, 3), use_batchnorm=False, if_transfer=False, if_local=True):
    axis = 3
    kernel_initializer = 'he_normal'
    origin_filters = 32
    weights = None
    model_path = os.path.join(sys.path[0], 'models', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    if if_transfer:
        weights = model_path if if_local else 'imagenet'
    weights = 'imagenet'
    vgg16 = VGG16(include_top=False, weights=weights, input_shape=input_size)
    for layer in vgg16.layers:
        layer.trainable = False

    input_img = Input(shape=input_size)  
    cross_domain_features = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(input_img)  
    cross_domain_features = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer)(cross_domain_features)   
    vgg16_input = vgg16(cross_domain_features)

    output = vgg16_input.layers[17].output

    up6 = layers.Conv2D(origin_filters * 8, 2, activation='relu', padding='same',
                        kernel_initializer=kernel_initializer)(
        layers.UpSampling2D(size=(2, 2))(output))
    merge6 = layers.concatenate([vgg16.layers[13].output, up6], axis=axis)

    conv6 = layers.Conv2D(origin_filters * 8, 3, activation='relu', padding='same',
                          kernel_initializer=kernel_initializer)(merge6)

    # 使用SWIM Transformer替换分割头
    transformer_input = layers.Conv2D(origin_filters * 8, 1)(conv6)  # 降维
    transformer_input = layers.Reshape((-1, origin_filters * 8))(transformer_input)  # 变形为合适的输入
    decoder_output = SWIMTransformer(embed_dim=origin_filters * 8, num_heads=8, ff_dim=256)(transformer_input,
                                                                                            training=True)
    decoder_output = layers.Reshape((conv6.shape[1], conv6.shape[2], origin_filters * 8))(decoder_output)  # 还原维度

    # 最后的输出层
    conv10 = layers.Conv2D(3, 1, activation='sigmoid')(decoder_output)
    model = tf.keras.Model(inputs=vgg16.input, outputs=conv10)

    return model

model = VGG16_unet_model(input_size=(288, 384, 3))
print(model.summary())

# 参数设置，命令行参数解析
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default="./dataset_2/dataset_2", required=False, help='path to dataset')
    parser.add_argument('--img_enhanced', default=False, help='image enhancement')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size')
    parser.add_argument('--image-size', default=(288, 384, 3), help='the (height, width, channel) of the input image to network')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
    # parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--model-save', default='./models/level3_model.h5', help='folder to output model checkpoints')
    parser.add_argument('--model-path', default='./models/level3_model.h5', help='folder of model checkpoints to predict')
    parser.add_argument('--outf', default="./test/test-level3", required=False, help='path of predict output')
    args = parser.parse_args(args=[])
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    return args


# 模型训练
def train_level3():
    args = get_parser()  # 获取参数
    train, train_label = load_image(args.data_root, "train", need_enhanced=args.img_enhanced)  # dataset为实际使用数据
    val, val_label = load_image(args.data_root, 'val', need_enhanced=args.img_enhanced)
    model = VGG16_unet_model(input_size=args.image_size, if_transfer=True, if_local=True)
    model.compile(optimizer=optimizers.Adam(lr=args.lr), loss=dice_coefficient_loss, metrics=[dice_coefficient])
    model_checkpoint = callbacks.ModelCheckpoint(args.model_path, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
    history = model.fit(train, train_label, batch_size=args.batch_size, epochs=args.niter, callbacks=[model_checkpoint],
                        validation_data=(val, val_label))
    plot_history(history, args.outf)

# 计算像素精度PA
def pixel_accuracy(label, predict):
    start_time = time.time()
    length, row, column, channels = label.shape
    true_pixel = 0
    all_pixels = length*row*column
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                if predict_cate == label_cate:
                        true_pixel = true_pixel + 1

    end_time = time.time()
    print("the pixel_accuracy is: " + str(true_pixel/all_pixels))
    print("compute pixel_accuracy use time: "+str(end_time-start_time))
    return true_pixel/all_pixels


# 计算均相似精度MPA
def mean_pixel_accuracy(label, predict, class_num = 4):
    start_time = time.time()
    length, row, column, channels = label.shape
    class_list = np.zeros(class_num)
    insaction_list = np.zeros(class_num)
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                for n in range(class_num):
                    if label_cate == n:
                        class_list[n] = class_list[n] + 1
                        if predict_cate == n:
                            insaction_list[n] = insaction_list[n] + 1
                        break
    end_time = time.time()
    mean_pixel_accuracy = 0
    for i in range(len(insaction_list)):
        if class_list[i] != 0:  # 检查除数是否为 0
            mean_pixel_accuracy += insaction_list[i] / class_list[i]
        else:
            # 可以选择跳过该项，或使用一个默认值，如 0
            print(f"Warning: class_list[{i}] is 0, skipping this item.")
            mean_pixel_accuracy += 0  # 或者设置一个合理的默认值

    mean_pixel_accuracy = mean_pixel_accuracy/class_num
    print("the mean pixel accuracy is: " + str(mean_pixel_accuracy))
    print("compute mean pixel accuracy use time: "+str(end_time-start_time))
    return mean_pixel_accuracy


# 计算均交并比mIoU
def compute_mIoU(label, predict, class_num = 4):
    start_time = time.time()
    length, row, column, channels = label.shape
    class_list = np.zeros(class_num)
    insaction_list = np.zeros(class_num)
    for i in range(length):
        for j in range(row):
            for m in range(column):
                predict_cate = category(predict[i, j, m, :])
                label_cate = category(label[i, j, m, :])
                for n in range(class_num):
                    if label_cate == n | predict_cate == n:
                        class_list[n] = class_list[n] + 1
                        if predict_cate == label_cate:
                            insaction_list[n] = insaction_list[n] + 1
    mIoU = 0
    for i in range(class_num):
        mIoU += insaction_list[i] / class_list[i]
    mIoU = mIoU / class_num
    end_time = time.time()
    print("the mIoU is: " + str(mIoU))
    print("compute_mIoU use time: "+str(end_time-start_time))
    return mIoU


def category(img):
    if img[0] >= 0.5:
        return 1
    elif img[1] >= 0.5:
        return 2
    elif img[2] >= 0.5:
        return 3
    else:
        return 0

def resize_test_images(test_images, target_size=(288, 384)):
    resized_images = []
    for img in test_images:
        resized_img = cv2.resize(img, (target_size[1], target_size[0]))  # 调整大小
        resized_images.append(resized_img)
    return np.array(resized_images)

def resize_predictions(predictions, original_size=(500, 574)):
    resized_predictions = []
    for pred in predictions:
        resized_pred = cv2.resize(pred, (original_size[1], original_size[0]))  # 调整大小
        resized_predictions.append(resized_pred)
    return np.array(resized_predictions)


def build_model(input_size=(288, 384, 3), use_batchnorm=False, if_transfer=False, if_local=True):
    """
    构建并返回一个使用VGG16作为基础网络并结合SWIM Transformer的U-Net模型
    """
    model = VGG16_unet_model(input_size=input_size, use_batchnorm=use_batchnorm,
                             if_transfer=if_transfer, if_local=if_local)

    # 编译模型，这里使用了常见的损失函数和优化器，你可以根据需要修改
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=dice_coefficient_loss,
                  metrics=[dice_coefficient])

    return model


# 模型测试
def predict_level3():
    args = get_parser()  # 获取参数
    test_img, test_label, test_name_list = load_image(args.data_root, "test", need_name_list=True,
                                                      need_enhanced=args.img_enhanced)
    # model = load_model(args.model_path, custom_objects={'dice_coefficient': dice_coefficient,
    #                                                     'dice_coefficient_loss': dice_coefficient_loss,
    #                                                     'SWIMTransformer': SWIMTransformer})
    model = build_model(input_size=(288, 384, 3), use_batchnorm=True, if_transfer=True)
    model.load_weights(args.model_path)
    # Step 1: 调整测试图片大小为模型输入的大小 (288x384)
    test_img_resized = resize_test_images(test_img, target_size=(288, 384))

    # Step 2: 使用模型进行预测
    result_resized = model.predict(test_img_resized)

    # Step 3: 将预测结果调整回原始大小 (500x574)
    result = resize_predictions(result_resized, original_size=(500, 574))

    dc = dice_coff(test_label, result)
    print("the dice coefficient is: " + str(dc))
    pixel_accuracy(test_label, result)
    mean_pixel_accuracy(test_label, result)
    compute_mIoU(test_label, result)
    for i in range(result.shape[0]):
        final_img = tensorToimg(result[i])
        ori_img = test_img[i]
        ori_gt = tensorToimg(test_label[i])

        plt.figure(figsize=(6, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(ori_img, cmap='gray')
        plt.axis('off')
        plt.text(x=50, y=-15, s="ori_image", ha='center', va='baseline',
                 fontdict=dict(fontsize=10, color="b", family='monospace', weight='bold'))
        plt.subplot(1, 3, 2)
        plt.imshow(ori_gt, cmap='gray')
        plt.axis('off')
        plt.text(x=50, y=-15, s="ori_gt", ha='center', va='baseline',
                 fontdict=dict(fontsize=10, color="b", family='monospace', weight='bold'))
        plt.subplot(1, 3, 3)
        plt.imshow(final_img, cmap='gray')
        plt.axis('off')
        plt.text(x=50, y=-15, s=f"predict", ha='center', va='baseline',
                 fontdict=dict(fontsize=10, color="b", family='monospace', weight='bold'))
        plt.text(x=50, y=255, s=f"dice_coff: {dc:2f}", ha='center', va='baseline',
                 fontdict=dict(fontsize=10, color="r", family='monospace', weight='bold'))
        # plt.show()
        plt.savefig(f"{args.outf}/{test_name_list[i]}")
        print(f"Save: {args.outf}/{test_name_list[i]}")
        plt.cla()
        plt.close("all")


if __name__ == "__main__":
    s_t = time.time()
    # train_level3()
    predict_level3()
    print("time:", time.time()-s_t)