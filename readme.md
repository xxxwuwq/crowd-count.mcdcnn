# crowd-count
## multi-column dilated cnn

## 目录结构及含义
````
facial-keypoints-detection
    ckpts: 用于存放模型文件
        'model_name': 根据指定的模型名称创建文件夹，用于存储该模型
    data: 用于存放数据集(案例为ShanghaiTech)
        ShanghaiTech:
            part_A_final:
                train_data:
                    ground_truth:
                    imgages:
                test_data:
                    ground_truth:
                    imgages:
    logs:
        'model_name': 根据指定的模型名称创建文件夹，用于存储该模型日志文件
    configs.py: 用于网络超参数及训练日志模型路径的设置等
    models.py: 网络结构文件
    train.py: 训练文件
    readme.md
````
## 网络结构
改进于MCNN的多列结构，使用空洞率为2的空洞卷积(除最后一层卷积层外)
`详细请见models.py`

