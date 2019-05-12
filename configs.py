#-*-coding:utf-8-*-


class Config(object):
    """
    网络超参数及相关路径配置
    """
    def __init__(self, model_name):
        self.model_name = model_name            # 模型名称 #
        self.batch_size = 1                     # batch size 为1 #
        self.lr = 1e-7                          # 学习率 #
        self.lr_decay_rate = 0.9                # 学习率衰变率 #
        self.lr_decay_step = 30000              # 学习率衰变步长 #
        self.momentum = 0.9                     # momentum优化器参数 #
        self.iter_num = 3000000                 # 迭代次数 #
        self.max_ckpt_keep = 50                 # 模型最多保存50个 #
        self.ckpt_router = './ckpts/' + self.model_name + r'/'  # 模型保存路径 #
        self.log_router = './logs/' + self.model_name + r'/'    # 训练日志保存路径 #
        self.snap = 25                          # 每snap个Epoch测试一次 #
    def display_configs(self):
        """
        打印配置信息
        :return:
        """

        msg = '''
        ------------ info of %s model -------------------
        batch size              : %s
        learing rate            : %f
        learing rate decay      : %f
        momentum                : %f
        iter num                : %s
        max ckpt keep           : %s
        ckpt router             : %s
        log router              : %s
        ------------------------------------------------
        ''' % (self.model_name, self.batch_size, self.lr, self.lr_decay_rate, self.momentum, self.iter_num, self.max_ckpt_keep, self.ckpt_router, self.log_router)
        print(msg)
        return msg


if __name__ == '__main__':
    configs = Config('MDCNN')
    configs.display_configs()