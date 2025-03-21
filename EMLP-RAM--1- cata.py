# fcnet.py
# MLP-stype model for continuous outputs

# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy
from sklearn.utils import shuffle
from prettytable import PrettyTable
from scipy.interpolate import make_interp_spline

import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px


# import local libraries
from network_interpret import StaticInterpret
from data_formatter import FormatDeliveries as Format
from data_formatter import Format as GeneralFormat

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Attention类
class Attention(nn.Module):
    """Attention mechanism."""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = F.softmax(Q @ K.transpose(-2, -1) / (K.size(-1) ** 0.5), dim=-1)
        return attention_scores @ V

# 这个类继承自torch.nn.Module，用于创建一个多层感知机模型
class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, output_size, return_embeddings=True):
        super().__init__()
        self.input_size = input_size
        hidden1_size = 500
        hidden2_size = 100
        hidden3_size = 20
        self.input2hidden = nn.Linear(input_size, hidden1_size)
        self.attention1 = Attention(hidden1_size)
        self.hidden2hidden = nn.Linear(hidden1_size, hidden2_size)
        self.attention2 = Attention(hidden2_size)
        self.hidden2hidden2 = nn.Linear(hidden2_size, hidden3_size)
        self.attention3 = Attention(hidden3_size)
        self.hidden2output = nn.Linear(hidden3_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.)
        self.return_embeddings = return_embeddings

    def forward(self, input):
        out = self.input2hidden(input)
        out = out + self.attention1(out)  # 残差连接
        em1 = out
        out = self.relu(out)
        out = self.dropout(out)
        em2 = out

        out = self.hidden2hidden(out)
        out = out + self.attention2(out)  # 残差连接
        em3 = out
        out = self.relu(out)
        out = self.dropout(out)
        em4 = out

        out = self.hidden2hidden2(out)
        out = out + self.attention3(out)  # 残差连接
        em5 = out
        out = self.relu(out)
        out = self.dropout(out)
        em6 = out

        output = self.hidden2output(out)

        if self.return_embeddings:
            return output, em1, em2, em3, em4, em5, em6
        else:
            return output


# 这个类不继承任何特定的库类，主要用于模型的训练、测试和评估

class ActivateNet:

    def __init__(self, epochs, deliveries=False):

        if deliveries:
            # specific dataset initialization and encoding
            file = 'titanic/train.csv'
            df = pd.read_csv(file)
            input_tensors = Format(file, 'Survived')
            # 调用sequential_tensors方法生成训练和验证的输入和输出张量
            self.input_tensors, self.output_tensors = input_tensors.sequential_tensors(training=True)
            self.validation_inputs, self.validation_outputs = input_tensors.sequential_tensors(training=False)
            self.n_letters = input_tensors.embedding_dim

        else:
            # general dataset initialization and encoding
            file = 'titanic/train.csv'
            form = GeneralFormat(file, 'Survived', ints_only=False)
            self.input_tensors, self.output_tensors = form.transform_to_tensors(training=True)
            self.validation_inputs, self.validation_outputs = form.transform_to_tensors(training=False)
            self.taken_ls = [form.n_taken for i in range(len(form.training_inputs.loc[0]))]
            self.n_letters = len(form.places_dict)

        print(len(self.input_tensors), len(self.validation_inputs))
        self.epochs = epochs
        output_size = 2
        input_size = len(self.input_tensors[0])
        self.model = MultiLayerPerceptron(input_size, output_size)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.biases_arr = [[], []]


    @staticmethod
    # 展示和计算一个PyTorch模型中可训练参数的数量
    def count_parameters(model):
        """
        Display the tunable parameters in the model of interest

        Args:
            model: torch.nn object

        Returns:
            total_params: the number of model parameters

        """
        # 使用PrettyTable库创建了一个表格，用于展示每个模块的名称和参数数量。这里的Modules和Parameters是表头。
        table = PrettyTable(['Modules', 'Parameters'])
        # 初始化一个变量total_params为0，用于累加模型中所有可训练参数的数量
        total_params = 0
        # 遍历模型的所有参数
        for name, parameter in model.named_parameters():
            # 检查参数是否需要梯度计算（即是否是可训练的参数）。如果参数不需要梯度，使用continue跳过当前循环。
            if not parameter.requires_grad:
                continue
            # 计算当前参数的元素数量，即参数中包含的标量值的总数
            param = parameter.numel()
            # 将当前参数的名称和数量添加到之前创建的表格中
            table.add_row([name, param])
            # 将当前参数的数量加到total_params变量上，累加总的参数数量
            total_params += param
        # 打印出包含所有可训练参数名称和数量的表格
        print (table)
        # 打印出可训练参数的总数量
        print (f'Total trainable parameters: {total_params}')
        # 返回可训练参数的总数量
        return total_params

    # 用于计算自定义的加权均方误差损失（MSE），根据问题的要求，当预测值低估真实值时，误差的成本是高估时的两倍
    # output和target，分别代表模型的预测值和真实值，都是torch.tensor类型。
    #
    # 这个weighted_mseloss函数在项目中用于自定义损失计算，特别适用于那些低估成本高于高估成本的场景。
    # 通过对低估情况施加更重的惩罚（损失加倍），它鼓励模型在预测时避免低估。
    def weighted_mseloss(self, output, target):
        """
        We are told that the true cost of underestimation is twice
        that of overestimation, so MSEloss is customized accordingly.

        Args:
            output: torch.tensor
            target: torch.tensor

        Returns:
            loss: float

        """
        # 判断预测值是否小于真实值，即是否发生了低估。
        if output < target:
            # 如果发生了低估，计算加权的均方误差损失，通过将预测值与真实值的差距乘以2并求平方，然后计算这些值的平均数来得到损失值。
            loss = torch.mean((2*(output - target))**2)
        # 如果预测值没有低估真实值（即预测值大于或等于真实值）
        else:
            # 计算标准的均方误差损失，通过求预测值与真实值差的平方的平均数。
            loss = torch.mean((output - target)**2)

        return loss

    # 这个weighted_l1loss函数实现了一种自定义的L1损失，该损失对于预测值低于真实值的情况赋予了双重权重。
    # 这是为了处理在某些应用场景中，低估真实值比高估有更严重的后果的情形。
    def weighted_l1loss(self, output, target):
        """
        Assigned double the weight to underestimation with L1 cost

        Args:
            output: torch.tensor
            target: torch.tensor

        Returns:
            loss: float
        """

        if output < target:
            loss = abs(2 * (output - target))

        else:
            loss = abs(output - target)

        return loss

    # 用于训练单个小批量(minibatch)数据的方法
    #
    # 这个函数train_minibatch的用途是在神经网络训练过程中，针对单个小批量(minibatch)的数据执行一次前向传播和反向传播，
    # 以此来更新模型的参数。它通过计算预测输出和真实目标之间的损失，使用梯度下降算法优化模型参数，从而减少模型的损失。
    # 这种按小批量更新模型参数的方法有助于提高模型训练的效率和收敛速度。
    def train_minibatch(self, input_tensor, output_tensor, minibatch_size):
        """
        Train a single minibatch

        Args:
            input_tensor: torch.Tensor object
            output_tensor: torch.Tensor object
            optimizer: torch.optim object
            minibatch_size: int, number of examples per minibatch
            model: torch.nn

        Returns:
            output: torch.Tensor of model predictions
            loss.item(): float of loss for that minibatch

        """
        # 将模型设置为训练模式，启用dropout和batch normalization等特性。
        self.model.train()
        # print(input_tensor.shape)
        # 通过模型传递输入张量input_tensor，获取模型的输出。这里使用了多个下划线（_）来忽略不需要的返回值。
        output, _, _, _, _, _, _ = self.model(input_tensor)
        # 将output_tensor（目标张量）重塑为与小批量大小相匹配的形状，每个小批量有1个输出。
        output_tensor = output_tensor.reshape(minibatch_size, 1)
        # 创建一个L1损失函数实例，用于计算预测值与实际值之间的差的绝对值的平均。
        loss_function = torch.nn.L1Loss()
        # 使用L1损失函数计算模型输出和目标张量之间的损失。
        # print(output.shape)
        # print(output_tensor.shape)
        loss = loss_function(output, output_tensor)
        # 清除（重置）模型参数的梯度，为新的梯度计算做准备。这是避免在多个小批量之间梯度累加的重要步骤。
        self.optimizer.zero_grad() # prevents gradients from adding between minibatches
        # 执行反向传播，根据损失计算模型参数的梯度
        loss.backward()
        # 对梯度进行裁剪，以防止梯度爆炸。这里将梯度的范数限制在0.3以内。
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
        # 根据计算得到的梯度更新模型的参数，这一步实现了模型的学习过程。
        self.optimizer.step()

        return output, loss.item()

    # 函数主要用于绘制模型预测值与真实值之间的对比图
    def plot_predictions(self, epoch_number):
        """
        Plots the model predictions (y-axis) versus the true output (x-axis)
        绘制模型预测(y轴)与真实输出(x轴)的关系
        Args:
            epoch_number: int,

        """
        # 将模型设置为评估模式，这样可以关闭dropout等只在训练时使用的功能，确保预测性能稳定
        self.model.eval() # switch to evaluation mode (silence dropouts etc.)
        # 初始化三个列表model_outputs, targets, origin_ids，分别用于存储模型的预测值、真实目标值和原始数据的索引
        model_outputs, targets, origin_ids = [], [], []
        # 在这个代码块内部，不计算梯度，这样可以加速预测过程并减少内存使用
        with torch.no_grad():
            total_error = 0
            for i in range(200):
                # 将输入和输出张量移动到指定的计算设备上
                input_tensor = self.validation_inputs[i].unsqueeze(0).to(device)
                output_tensor = self.validation_outputs[i].to(device)

                model_output, _, _, _, _, _, _ = self.model(input_tensor)
                model_outputs.append(float(model_output))
                origin_ids.append(i)
                targets.append(float(output_tensor))

        # 将模型的预测输出和真实输出值转换为浮点数，然后分别添加到model_outputs和targets列表中
        # 使用matplotlib.pyplot（别名plt）和plotly.express（别名px）绘制散点图，将真实值作为x轴，
        # 预测值作为y轴，通过不同的颜色表示不同的数据点
        plt.scatter(targets, model_outputs, c=origin_ids, cmap='hsv', s=1.5)
        fig = px.scatter(x=targets,
                         y=model_outputs,
                         color=origin_ids,
                         color_continuous_scale=["red", "orange","yellow", "green","blue", "purple", "violet"])

        fig.update_traces(textposition='top center')
        # fig.show()

        # _, _, rval, _, _ = scipy.stats.linregress([float(i) for i in self.validation_outputs], model_outputs)
        # print (f'R2 value: {rval**2}')
        # plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
        # plt.show()
        # plt.savefig('filename.png')

        # 调整图形布局，使之紧凑
        plt.tight_layout()
        # 保存图形为PNG格式，文件名包含训练的epoch编号
        plt.savefig('regression_emlp_ram_cata/regression{0:04d}_emlp_ram_cata.png'.format(epoch_number), dpi=400)
        # 关闭图形，释放资源
        plt.close()

        return

    # 这个好像没画出图来
    # 将模型的偏置项（biases）通过散点图的形式进行可视化
    #
    # 偏置项（bias）是神经网络中的一个参数，用于调整输出以外的一个固定偏移量。
    # 每个神经元在计算加权输入和激活函数之前会加上这个偏置项。
    # 在上下文中提到的self.model.hidden2hidden.bias[:6]表示的是模型中从一个隐藏层到另一个隐藏层的权重矩阵中前6个偏置项。
    # 这些偏置项对模型的决策边界有重要影响，因为它们可以调整神经元的激活阈值。通过观察这些偏置项的变化，可以了解模型在训练过程中的学习动态
    def plot_biases(self, index):
        """
        Image model biases as a scatterplot

        Args:
            index: int

        Returns:
            None
        """
        # 将模型设置为评估模式，这通常意味着关闭dropout等只在训练时使用的特性
        self.model.eval()
        # 从模型中的某个隐藏层到另一个隐藏层的权重中，获取前6个偏置项，并将它们从PyTorch张量转换为NumPy数组，以便进行处理和可视化
        arr = self.model.hidden2hidden.bias[:6].detach().numpy()
        # 将提取的偏置项分组（每组两个）并分别添加到三个不同的列表中，用于后续的绘图
        self.biases_arr[0].append([arr[0], arr[1]])
        self.biases_arr[1].append([arr[2], arr[3]])
        self.biases_arr[2].append([arr[4], arr[5]])
        plt.style.use('dark_background')
        # 接下来的三行plt.plot(...)命令分别绘制了三组偏置项的散点图，使用不同的颜色和标记来区分
        plt.plot([i[0] for i in self.biases_arr[0]], [i[1] for i in self.biases_arr[0]], '^', color='white', alpha=0.7, markersize=0.1)
        plt.plot([i[0] for i in self.biases_arr[1]], [i[1] for i in self.biases_arr[1]], '^', color='red', alpha=0.7, markersize=0.1)
        plt.plot([i[0] for i in self.biases_arr[2]], [i[1] for i in self.biases_arr[2]], '^', color='blue', alpha=0.7, markersize=0.1)
        # 确保坐标轴是开启状态，以便在图像中显示
        plt.axis('on')
        # 保存绘制的图像到文件系统，文件名中包含传入的index参数，以便于跟踪和比较不同时间点的偏置变化
        plt.savefig('Biases_{0:04d}.png'.format(index), dpi=400)
        plt.close()

        return

    # 这个图也没画出来
    # 函数用于将模型中某一层的权重以热图的形式进行可视化
    def heatmap_weights(self, index):
        """
        Plot model weights of one layer as a heatmap

        Args:
            index: int

        Returns:
            None

        """
        self.model.eval()
        # 获取名为hidden2hidden2的层的权重，并使用detach()方法将其从当前计算图中分离，这样不会在后续操作中影响梯度计算
        arr = self.model.hidden2hidden2.weight.detach()
        # 将权重张量重塑为2000行1列的形状
        arr = torch.reshape(arr, (2000, 1))
        # 从重塑后的张量中取出前1936（44*44）个元素
        arr = arr[:44*44]
        # 将这1936个元素重塑为44行44列的二维张量，准备以热图形式展示
        arr = torch.reshape(arr, (44, 44))
        # 将PyTorch张量转换为NumPy数组，以便使用matplotlib进行绘图
        arr = arr.numpy()
        # 使用matplotlib的imshow函数绘制热图，interpolation='spline16'设置插值方式，aspect='auto'自动调整长宽比，cmap='inferno'设置颜色映射。
        plt.imshow(arr, interpolation='spline16', aspect='auto', cmap='inferno')
        plt.style.use('dark_background')
        plt.axis('off')
        # 保存热图到文件，文件名包含传入的index参数，dpi=400设置图像分辨率，bbox_inches='tight', pad_inches=0去除周围空白边距
        plt.savefig('heatmap_weight_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()
        return

    # 以热图的形式可视化模型中的偏置项
    def heatmap_biases(self, index):
        """
        Plot model biases as a rectangular heatmap

        Args:
            index: int

        Returns:
            None

        """
        self.model.eval()
        arr = self.model.hidden2hidden.bias.detach().numpy()

        # convert to 10x10 array
        arr2 = []
        j = 0
        while j in range(len(arr)):
            ls = []
            while len(ls) < 10 and j < len(arr):
                ls.append(round(arr[j], 3))
                j += 1
            arr2.append(ls)

        arr2 = np.array(arr2)
        plt.imshow(arr2, interpolation='spline16', aspect='auto', cmap='inferno')
        for (y, x), label in np.ndenumerate(arr2):
            plt.text(x, y, '{:1.3f}'.format(label), ha='center', va='center', fontsize=6)

        plt.axis('off')
        plt.rcParams.update({'font.size': 6})
        # plt.style.use('dark_background')
        plt.savefig('heatmap_biases_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()

        return

    def plot_embedding(self, index=0):
        """
        Generates a scatterplot of all pairs of embeddings versus the input
        distance of significance (ie the output for a control experiment)
        生成所有嵌入对与输入的散点图
        显着性距离（即对照实验的输出）
        Args:
            None

        Returns:
            None (saves .png)
        """

        for k in range(0, 1):
            self.model.eval() # switch to evaluation mode (silence dropouts etc.)
            number_of_examples = 200
            actual_arr, embedding_arr, input_arr = [], [], []
            for i in range(number_of_examples):
                input_tensor = self.validation_inputs[i].unsqueeze(0).to(device)
                output_tensor = self.validation_outputs[i].to(device)

                model_output, embedding = self.model(input_tensor)[0], self.model(input_tensor)[k]
                actual_arr.append(float(output_tensor))
                embedding_arr.append(embedding)
                input_arr.append(input_tensor)

            actual_distances, embedding_distances, input_distances = [], [], []
            origin_id, target_id = [], []
            for i in range(len(embedding_arr) - 1):
                for j in range(i+1, len(embedding_arr)):
                    actual_distances.append(np.abs(actual_arr[j] - actual_arr[i]))
                    embedding_distance = torch.sum(torch.abs(embedding_arr[j] - embedding_arr[i])).cpu().detach().numpy()
                    input_distance = torch.sum(torch.abs(input_arr[j][14*15:31*15] - input_arr[i][14*15:31*15])).cpu().detach().numpy()
                    embedding_distances.append(embedding_distance)
                    input_distances.append(input_distance)
                    origin_id.append(i)
                    target_id.append(j)

            plt.rcParams.update({'font.size': 17})
            plt.scatter(actual_distances, embedding_distances, c=origin_id, cmap='hsv', s=0.3)
            # plt.scatter(actual_distances, embedding_distances, s=0.3)
            plt.xlabel('Actual Distance')
            plt.ylabel('Embedding Distance')
            plt.tight_layout()
            plt.savefig('embedding_emlp_ram_cata/embedding{0:04d}_emlp_ram_cata'.format(k), dpi=350)
            plt.close()

            fig = px.scatter(x=actual_distances,
                             y=embedding_distances,
                             color=origin_id,
                             hover_data=[origin_id, target_id],
                             color_continuous_scale=["red", "orange","yellow", "green","blue", "purple", "violet"])
            fig.update_traces(textposition='top center')
            # fig.show()

            plt.scatter(input_distances, embedding_distances, s=0.3)
            plt.xlabel('Input Distance')
            plt.ylabel('Embedding Distance')
            plt.rcParams.update({'font.size': 18})
            plt.savefig('input_embedding_emlp-ram_cata/input_embedding_emlp-ram_cata'.format(index), dpi=390)
            plt.close()
        return

    # 用于训练神经网络
    def train_model(self, minibatch_size=128):
        """
        Train the neural network.

        Args:
            model: MultiLayerPerceptron object
            optimizer: torch.optim object
            minibatch_size: int

        Returns:
            None

        """

        self.model.train()
        epochs = self.epochs
        count = 0
        losses = []  # 初始化一个列表来保存每个批次的损失值
        for epoch in range(epochs):
            pairs = [[i, j] for i, j in zip(self.input_tensors, self.output_tensors)]
            random.shuffle(pairs)
            input_tensors = [i[0] for i in pairs]
            output_tensors = [i[1] for i in pairs]
            total_loss = 0

            for i in range(0, len(input_tensors) - minibatch_size, minibatch_size):
                # print (count)
                input_batch = torch.stack(input_tensors[i:i + minibatch_size]).to(device)
                output_batch = torch.stack(output_tensors[i:i + minibatch_size]).to(device)

                # skip the last batch if it is smaller than the other batches
                if len(input_batch) < minibatch_size:
                    break

                output, loss = self.train_minibatch(input_batch, output_batch, minibatch_size)
                total_loss += loss
                losses.append(loss)  # 保存当前批次的损失值
                # if count % 100 == 0:
                    # self.plot_embedding(index=count // 100)
                    # interpret = StaticInterpret(self.model, self.validation_inputs, self.validation_outputs,self.embedding_dim)
                    # self.plot_predictions(count // 25)
                # if count % 100 == 0:
                # 	self.plot_embedding(index=count//100)
                    # interpret = StaticInterpret(self.model, self.validation_inputs, self.validation_outputs)
                    # self.plot_predictions(count//25)
                    # interpret.heatmap(count, method='combined')
                count += 1

            print (f'Epoch {epoch} complete: {total_loss} loss')
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Loss per Batch')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.title('Loss per Batch During Training')
        plt.savefig('loss_emlp_ram_cata/loss_emlp_ram_cata.png', dpi=400)
        plt.legend()
        plt.show()
        return

    @torch.no_grad()
    def test_model(self, n_taken=2000):
        """
        Test the model using a validation set

        kwargs:
            n_taken: int, number of rows used to test model

        Returns:
            None (prints test results)

        """
        print("0")
        self.model.eval() # switch to evaluation mode (silence dropouts etc.)
        loss = torch.nn.L1Loss()

        total_error = 0

        for i in range(len(self.validation_inputs)):
            input_tensor = self.validation_inputs[i].unsqueeze(0).to(device)
            output_tensor = self.validation_outputs[i].to(device)

            # print(input_tensor.shape)
            model_output, *_ = self.model(input_tensor)

            total_error += loss(model_output, output_tensor).item()

        print (f'Test Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')

        total_error = 0
        for i in range(len(self.input_tensors[:n_taken])):
            input_tensor = self.input_tensors[i].unsqueeze(0)
            output_tensor = self.output_tensors[i]
            model_output, *_ = self.model(input_tensor.to(device))
            total_error += loss(model_output, output_tensor.to(device)).item()

        print (f'Training Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')
        # 下面是加的
        minibatch_size = 1
        with torch.no_grad():
            correct, count = 0, 0
            for i in range(0, len(self.validation_inputs), minibatch_size):
                input_batch = torch.stack(self.validation_inputs[i:i + minibatch_size])
                output_batch = torch.stack(self.validation_outputs[i:i + minibatch_size])
                input_tensor = input_batch.to(device)
                output_tensor = output_batch.reshape(minibatch_size).to(device)
                model_output, *_ = self.model(input_tensor)
                correct += torch.sum(torch.argmax(model_output, dim=1) == output_tensor)
                count += minibatch_size
        print(f'Test Accuracy: {correct / count}')


        return

if __name__ == '__main__':
    network = ActivateNet(200)
    network.train_model()
    network.test_model()
    # torch.save(network.model.state_dict(), 'model_weights')
    # network.model.load_state_dict(torch.load('model_weights'))
    # network.heatmap_biases(0)
    # network.plot_embedding() # 这个也是
    # network.plot_predictions(0)# 这个也是

    # network.heatmap_weights(2)
    # network.heatmap_biases(2)
    # network.plot_biases(200)

    # Note that interpretations are not compatible with embedding information
    # input_tensors, output_tensors = network.validation_inputs, network.validation_outputs
    # model = network.model.to(device)
    # interpret = StaticInterpret(model, input_tensors, output_tensors, network.embedding_dim)
    # interpret.readable_interpretation(0, method='occlusion', aggregation='max')
    # interpret.heatmap(0, method='occlusion')

    # interpret.readable_interpretation(1, method='gradientxinput', aggregation='max')
    # interpret.heatmap(1, method='gradientxinput')

    # interpret.readable_interpretation(2, method='combined', aggregation='max')
    # interpret.heatmap(2, method='combined')






