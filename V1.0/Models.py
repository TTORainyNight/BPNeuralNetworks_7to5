import numpy as np
import matplotlib.pyplot as plt

class DClassifier:
    #7对5分类神经网络，具体细节请查看调用实例。

    #参数调整：在Train()方法中进行调整，一般来说，注释中不存在“to do”的参数，不建议调整。
    #参数调整：在Train()方法中进行调整，调整神经网络参数中，注释带有“to do”的参数，使得训练模型符合预期。

    #输入接口说明：
    #Operation：动作控制，参数为'train'时，执行训练操作，并保存模型；
    #Operation：动作控制，参数为'check'时，执行读取模型，检测操作；
    #Operation：参数不允许设置'train'与'check'之外的其它值；
    #InputData：训练操作时，此参数为训练集。参数类型为n*8的多维numpy数组，对于每行，前7位为数据，末位为标签；
    #InputData：检测操作时，此参数为检测目标。参数类型为1*7的一维numpy数组；
    #CheckData：训练操作时，此参数为检测集。参数类型与InputData参数应保持一致；
    #CheckData：检测操作时，此参数无意义，应当置为0或'null'；
    #Model_dir：训练操作时，代表模型的存放位置。参数类型为字符串，标准路径，例如："B:\\temp\\"；
    #Model_dir：检测操作时，代表模型的读取位置。参数类型与训练保持一致。

    #方法说明：
    #Run()：设置参数后，调用此方法以获取正确的输出。

    #输出接口说明：
    #训练操作时，你可以在Model_dir目录下找到四个模型文件，它们是：W1.npy b1.npy W2.npy b2.npy
    #Results：执行检测操作时，此参数存放模型对于InputData的预测结果。参数类型为int变量，范围是0-4

    #输入
    def __init__(self,InputData,CheckData,Operation,Model_dir):
        self.X = InputData
        self.CheckData = CheckData
        self.Operation = Operation
        self.Results = 'null'
        self.Model_dir = Model_dir
    #运行
    def Run(self):
        if self.Operation == 'train':
            self.Train()
        elif self.Operation == 'check':
            self.Check()
        else:
            print("DClassifier: Wrong Input!")
    #激活函数
    def softmax(self,x):
        return 1 / (1 + np.exp(-x))
    def softmax_derivative(self,x):
        return self.softmax(x) * (1 - self.softmax(x))
    #训练
    def Train(self):
        #导入
        X = self.X
        y = X[:, -1]
        X = X[:, :-1] / 100
        #神经网络参数
        input_size = 7 #输入层节点
        hidden_size = 13 #隐藏层节点 to do
        output_size = 5 #输出层节点
        learning_rate = 0.01 #学习率 to do
        epochs = 10000 #迭代次数 to do
        #初始化
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.random.randn(1, hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.random.randn(1, output_size)
        losses = []
        #训练
        y_onehot = np.eye(output_size)[y]
        for i in range(epochs):
            Z1 = X.dot(W1) + b1
            A1 = self.softmax(Z1)
            Z2 = A1.dot(W2) + b2
            A2 = self.softmax(Z2)
            loss = -np.mean(np.log(A2) * y_onehot)
    
            dZ2 = A2 - y_onehot
            dW2 = A1.T.dot(dZ2)
            db2 = np.sum(dZ2, axis=0)
            dZ1 = dZ2.dot(W2.T) * self.softmax_derivative(Z1)
            dW1 = X.T.dot(dZ1)
            db1 = np.sum(dZ1, axis=0)

            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            #输出信息
            losses.append(loss)
            if (i + 1) % 10 == 0:
                print(f"Epoch {i + 1}, loss = {loss:.4f}")
        #画图
        plt.plot(losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
        #评估
        check_num =self.CheckData
        check_label = check_num[:, -1]
        check_num = check_num[:, :-1] / 100
        right = 0
        for i in range(check_num.shape[0]):
            tem = check_num[i]
            Z1 = tem.dot(W1) + b1
            A1 = self.softmax(Z1)
            Z2 = A1.dot(W2) + b2
            A2 = self.softmax(Z2)
            if check_label[i] == np.argmax(A2):
                right = right + 1
        #输出保存
        print ('Successful----Model accuracy: {:.4%}'.format (right / check_num.shape[0]))
        np.save(self.Model_dir + "W1.npy", W1)
        np.save(self.Model_dir + "b1.npy", b1)
        np.save(self.Model_dir + "W2.npy", W2)
        np.save(self.Model_dir + "b2.npy", b2)
    #预测
    def Check(self):
        #获取模型
        W1 = np.load(self.Model_dir + "W1.npy")
        b1 = np.load(self.Model_dir + "b1.npy")
        W2 = np.load(self.Model_dir + "W2.npy")
        b2 = np.load(self.Model_dir + "b2.npy")
        #检测输出
        goal = self.X
        goal = goal / 100
        Z1 = goal.dot(W1) + b1
        A1 = self.softmax(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = self.softmax(Z2)
        self.Results = int(np.argmax(A2))