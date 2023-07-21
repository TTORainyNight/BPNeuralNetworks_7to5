import time
from tensorflow import keras as ks
import numpy as np
from keras import layers
from keras.datasets import mnist as mt
import matplotlib.pyplot as plt

'''
给出两个API调用示例，分别对应训练与评估：
train = {"Option": "train" ,
        "Train_Data": np.array([[90, 91, 93, 95, 93, 99, 89，1]]) ,
        "Check_Data": np.array([[90, 91, 93, 95, 93, 99, 89，1]]) ,
        "Study_Rate": 0.02 , "N": 800 , "File": r"C:\Temp\data.h5" ,
        "Visable": True , "TimeOut": 5}
evaluate = {"Option": "evaluate" ,
        "File": r"C:\Temp\data.h5" ,
        "Visable": False ,
        "Eva_Data": np.array([[90, 91, 93, 95, 93, 99, 89]])}
'''
class DClassifier():
    # 7对5分类神经网络，具体细节请查看调用实例。

    # API接口说明--通用参数：
    # Option：动作控制，参数为'train'时，执行训练操作，并保存模型；
    # Option：动作控制，参数为'evaluate'时，执行读取模型，评估操作，不允许设置'train'与'evaluate'之外的其它值；
    # File：指定模型文件，字符串文件路径，需要指定到.h5文件，训练时为模型保存位置，评估时为模型读取位置；
    # Visable：过程可视化，布尔型，只能为True/False。控制训练与评估的详细过程是否展示；
    # TimeOut：可视化训练开始前的延时，Visable为False时，不需要此参数。单位为秒，非负整数。

    # 训练参数：
    # Train_Data：训练集，参数类型为n*8的二维numpy数组，对于每行，前7位为数据，末位为标签；
    # Check_Data：检测集，参数类型与Train_Data参数应保持一致；
    # Study_Rate：学习率，0.0-1.0之间的小数；
    # N：迭代次数，非负整数；
    # Result：无需指定，这是输出参数。存放一个数组，[0]是保存时loss值，[1]是检测集准确率

    # 评估参数
    # Eva_Data：训练集，参数类型为n*7的二维numpy数组；
    # Result：无需指定，这是输出参数。存放一个array数组，[0]是预测值

    def __init__(self, InputData):
        self.ID = InputData
        if InputData["Visable"]:
            print("7对5BP神经网络，已读取到如下参数：\n---------------------")
            print(InputData)
        self.run()

    #操作导航
    def run(self):
        goal = self.ID["Option"]
        if goal == 'train':
            self.train()
        elif goal == 'evaluate':
            self.evaluate()
        else:
            print("DClassifier: Option Input Wrong Input!")

    #训练
    def train(self):
        visable = self.ID["Visable"]
        if visable:
            timeout = self.ID["TimeOut"]
            print(f"\n-----正在准备训练\n------------------------------------\n\n-----延时{timeout}秒\n------------------------------------\n")
            time.sleep(timeout)
            print("-----正在进行数据预处理 … …")

        #数据读取、预处理
        tdata = self.ID["Train_Data"]
        cdata = self.ID["Check_Data"]
        mdir = self.ID["File"]
        tlabel = tdata[:, -1]
        tdata = tdata[:, :-1] / 100
        clabel = cdata[:, -1]
        cdata = cdata[:, :-1] / 100
        
        #构建模型
        if visable:
            print("-----正在构建模型 … …")
        model = ks.Sequential() 
        model.add(layers.Flatten())
        model.add(layers.Dense(12, activation='relu'))
        model.add(layers.Dense(12, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))
        ss = self.ID["Study_Rate"]
        model.compile(optimizer=ks.optimizers.Adam(learning_rate=ss), loss = 'sparse_categorical_crossentropy', metrics=['acc'])
        verbose_index = 0
        if visable:
            print("-----模型构建完毕 … …")
            verbose_index = 1
            print("-----开始训练 … …")

        #训练模型
        history = model.fit(tdata, tlabel, epochs = self.ID["N"], verbose = verbose_index)
        if visable:
            plt.plot(history.history['loss'])
            plt.xlabel('epoch')
            plt.ylabel('loss')
        self.ID["Result"] = model.evaluate(cdata, clabel, verbose = verbose_index)
        if visable:
            print("-----模型训练完毕，以下是检测结果：")
        print("       ", self.ID["Result"])
        if visable:
            print("\n-----已存放至Result中 … …\n-----准备保存模型 … …")
        model.save(mdir)
        if visable:
            print(f"-----模型已保存至：{mdir}\n-----本次训练结束")
            plt.show()
    
    #评估
    def evaluate(self):
        #数据读取
        visable = self.ID["Visable"]
        if visable:
            timeout = self.ID["TimeOut"]
            print(f"\n-----正在准备评估\n--------------------------------\n\n-----延时{timeout}秒\n------------------------------------\n")
            time.sleep(timeout)
            print("-----正在进行数据预处理 … …")
        edata = self.ID["Eva_Data"]
        mdir = self.ID["File"]
        edata = edata / 100

        #模型读取
        if visable:
            print(f"-----正在读取模型{mdir} … …")
        model = ks.models.load_model(mdir)

        #预测评估
        verbose_index = 0
        if visable:
            verbose_index = 1
            print("-----开始进行评估 … …")
        self.ID["Result"] = np.argmax(model.predict(edata, verbose = verbose_index), axis= 1)
        if visable:
            print("-----评估结果已存放至Result中，运行结束 … …")
