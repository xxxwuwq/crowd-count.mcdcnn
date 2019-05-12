# -*-coding:utf-8-*-
import time
from tools import Tool
from models import Model
from configs import Config
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm

np.set_printoptions(threshold=np.inf)


def train(dataset='A'):
    # 训练数据集路径
    img_root_dir = './data/ShanghaiTech/part_' + dataset + '_final/train_data/images/'
    gt_root_dir = './data/ShanghaiTech/part_' + dataset + '_final/train_data/ground_truth/'
    # 测试数据集路径
    test_img_dir = './data/ShanghaiTech/part_' + dataset + '_final/test_data/images/'
    test_gt_dir = './data/ShanghaiTech/part_' + dataset + '_final/test_data/ground_truth/'

    img_file_list = os.listdir(img_root_dir)
    # gt_file_list = os.listdir(gt_root_dir)
    test_img_list = os.listdir(test_img_dir)
    # test_gt_list = os.listdir(test_gt_dir)

    model_name = 'mcdcnn_SHTech_' + dataset
    cfg = Config(model_name)
    cfg.lr = 1e-4
    tool = Tool()
    model = Model()
    network = model.mcdcnn()
    INPUT, GT_DMP, GT_CNT, EST_DMP, EST_CNT = network['INPUT'], network['GT_DMP'], network['GT_CNT'], \
                                              network['EST_DMP'], network['EST_CNT']

    loss = model.losses(GT_DMP, GT_CNT, EST_DMP, EST_CNT)

    # 学习率衰变设置
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(cfg.lr, global_step=global_step, decay_steps=cfg.lr_decay_step,
                                    decay_rate=cfg.lr_decay_rate, staircase=True)
    # 优化器
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step)

    # 模型保存设置
    saver = tf.train.Saver(max_to_keep=cfg.max_ckpt_keep)
    ckpt = tf.train.get_checkpoint_state(cfg.ckpt_router)

    # 创建会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 加载模型
        if ckpt and ckpt.model_checkpoint_path:
            print('load model')
            saver.restore(sess, ckpt.model_checkpoint_path)
        # 训练日志文件路径设置
        if not os.path.exists(cfg.log_router):
            os.makedirs(cfg.log_router)
        log = open(cfg.log_router + 'train' + r'.logs', mode='a+', encoding='utf-8')
        # 迭代训练
        num = len(img_file_list)
        # 开始训练
        for i in tqdm(range(cfg.iter_num)):
            # 随机打乱
            np.random.shuffle(img_file_list)
            for j in tqdm(range(num)):
                img_path = img_root_dir + img_file_list[j]
                gt_path = gt_root_dir + 'GT_' + img_file_list[j].split(r'.')[0]
                img, dmp, cnt = tool.read_train_data(img_path, gt_path, use_knn=True)
                feed_dict = {INPUT: img, GT_DMP: dmp, GT_CNT: cnt}

                _, est_dmp, est_cnt, train_loss = sess.run([optimizer, EST_DMP, EST_CNT, loss], feed_dict=feed_dict)
                format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                format_str = 'step %d, loss = %.3f, gt = %d, inference = %.3f'
                log_line = format_time, img_file_list[j], format_str % ((i * num + j), train_loss, cnt, est_cnt)
                print(log_line)
                log.writelines(str(log_line) + '\n')
                sess.run(lr, feed_dict={global_step: i * num + j})

            if i % cfg.snap == 0:
                # 保存模型
                saver.save(sess, cfg.ckpt_router + '/v1', global_step=i)
                # 进行测试
                print('testing', i, '> > > > > > > > > > > > > > > > > > >')
                total_mae = 0.0
                total_mse = 0.0
                num = len(test_img_list)
                log_line = ''
                for k in tqdm(range(num - 2, num)):
                    img_path = test_img_dir + test_img_list[k]
                    gt_path = test_gt_dir + 'GT_' + test_img_list[k].split(r'.')[0]

                    img, dmp, cnt = tool.read_test_data(img_path, gt_path, use_knn=True)

                    feed_dict = {INPUT: img, GT_DMP: dmp, GT_CNT: cnt}

                    est_cnt, test_loss = sess.run([EST_CNT, loss], feed_dict=feed_dict)

                    format_time = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                    format_str = 'step %d, joint loss = %.3f, gt = %d, inference = %.3f'
                    line = str(format_time + ' ' + test_img_list[k] + ' ' + format_str %
                               (k, test_loss, cnt, est_cnt) + '\n')
                    log_line += line
                    total_mae += tool.mae_metrix(cnt, est_cnt)
                    total_mse += tool.mse_metrix(cnt, est_cnt)
                    print(line)

                avg_mae = total_mae / num
                avg_mse = pow(total_mse / num, 0.5)
                result = str('_MAE_%.5f_MSE_%.5f' % (avg_mae, avg_mse))
                test_log = open(cfg.log_router + 'test_' + str(i) + result + r'.logs', mode='a+', encoding='utf-8')
                test_log.writelines(result + '\n')
                test_log.writelines(str(log_line) + '\n')
                test_log.close()
        log.close()


if __name__ == '__main__':
    train()
