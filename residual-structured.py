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
# from data_formatter import FormatDeliveries as Format
# from data_formatter import Format as GeneralFormat

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 这个类定义了一个标准的残差块，通常用于构建深度神经网络，帮助缓解梯度消失问题，增强模型在深层网络中的学习能力。
class ResidualBlock(nn.Module):
    """定义一个残差块，包括两个线性层和一个残差连接"""

    def __init__(self, input_size, output_size, activation_fn=nn.ReLU, dropout_rate=0.05):
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(output_size, output_size)
        self.activation = activation_fn()
        self.dropout = nn.Dropout(dropout_rate)
        self.batchnorm1 = nn.BatchNorm1d(output_size)
        self.batchnorm2 = nn.BatchNorm1d(output_size)

        # 如果输入和输出大小不一致，需要一个适配层来匹配它们
        self.match_dimension = input_size != output_size
        if self.match_dimension:
            self.dimension_adapter = nn.Linear(input_size, output_size)

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.batchnorm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.linear2(out)
        out = self.batchnorm2(out)

        # 如果输入和输出的维度不匹配，通过dimension_adapter调整identity的维度
        if self.match_dimension:
            identity = self.dimension_adapter(identity)

        out += identity  # 残差连接
        out = self.activation(out)

        return out


class DeepResidualMLP(nn.Module):
    """构建更深的MLP，使用残差块"""

    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 256, 128], return_embeddings=True):
        super().__init__()
        self.return_embeddings = return_embeddings
        layers = []
        in_features = input_size

        for hidden_size in hidden_sizes:
            layers.append(ResidualBlock(in_features, hidden_size))
            in_features = hidden_size

        self.residual_blocks = nn.Sequential(*layers)
        self.final_linear = nn.Linear(in_features, output_size)

    def forward(self, x):
        embeddings = []

        for block in self.residual_blocks:
            x = block(x)
            embeddings.append(x)

        output = self.final_linear(x)

        if self.return_embeddings:
            return output, *embeddings
        else:
            return output


# 这个类不继承任何特定的库类，主要用于模型的训练、测试和评估
class Format:

	def __init__(self, file, training=True):

		df = pd.read_csv(file)
		# 接下来，这行代码对 DataFrame df 中的每个元素应用一个函数，该函数的作用是检查元素是否为字符串 'nan'（不区分大小写），如果是，则将该元素替换为空字符串 ''；否则，保留元素的原始值。这里使用了 applymap 方法，它适用于 DataFrame 的每个元素。
		#
		# lambda x: '' if str(x).lower() == 'nan' else x 是一个匿名函数，用于定义替换规则。str(x).lower() == 'nan' 将元素转换为字符串并转换为小写，然后检查是否等于 'nan'。
		# 这种处理对于清洗数据很有用，特别是在处理缺失值时，某些情况下你可能希望将它们标记为特定的值（在这个例子中是空字符串）。
		df = df.applymap(lambda x: '' if str(x).lower() == 'nan' else x)
		df = df[:20000]

		# 这行代码计算 df 中 'Elapsed Time' 列的长度，即这一列（或说这个序列）中元素的数量
		length = len(df['Elapsed Time'])
		# 这行代码定义了一个列表 self.input_fields，它包含了一系列字符串，每个字符串都是 DataFrame df 中一列的名称。
		self.input_fields = ['Store Number',
							'Market',
							'Order Made',
							'Cost',
							'Total Deliverers',
							'Busy Deliverers',
							'Total Orders',
							'Estimated Transit Time',
							'Linear Estimation']

		if training:
			df = shuffle(df)# 这行代码使用 shuffle 函数对 DataFrame df 中的行进行随机重排,这对于很多机器学习任务非常重要，特别是在划分训练集和测试集之前，可以帮助减少模型训练过程中的偏差
			df.reset_index(inplace=True)# 放弃旧的索引并引入一个从 0 开始的新索引

			# 80/20 training/validation split
			split_i = int(length * 0.8)
			# 前80%是训练集
			training = df[:][:split_i]
			self.training_inputs = training[self.input_fields]# self.input_fields 是一个列表，包含了你想作为模型输入的列名。通过 training[self.input_fields]，从 training DataFrame 中选取这些指定的列，作为训练输入。这意味着 self.training_inputs 将只包含 input_fields 列中的数据，这些通常是特征列，用于训练模型。
			self.training_outputs = [i for i in training['positive_two'][:]]# 这行代码使用列表推导式从 training DataFrame 中提取 'positive_control' 列的所有值作为训练输出。这里 'positive_control' 应该是包含标签或目标变量的列。列表推导式 for i in training['positive_control'][:] 遍历 'positive_control' 列中的每个元素，将其添加到列表中，然后将这个列表赋值给 self.training_outputs。使用 [:] 是多余的，直接使用 training['positive_control'] 就可以达到同样的目的。

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.validation_inputs = validation[self.input_fields]
			self.validation_outputs = [i for i in validation['positive_two'][:]]
			self.validation_inputs = self.validation_inputs.reset_index()

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify


	# 函数的目的是将指定行的数据转换为一个结构化的字符串表示
	def stringify_input(self, index, training=True):
		# index: 这是一个整数，指定了要转换为字符串的数据行的索引
		"""
		Compose array of string versions of relevant information in self.df
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""
		# 结构化 taken_ls = [4, 1, 8, 5, 3, 3, 3, 4, 4] 定义了每个字段应该取的字符数。例如，第一个字段取前4个字符，第二个字段只取第一个字符，依此类推
		taken_ls = [4, 1, 8, 5, 3, 3, 3, 4, 4]
		# string_arr = [] 初始化一个空列表，用于存储转换后的字符串片段。
		string_arr = []
		# 根据 training 参数的值，函数决定是从 self.training_inputs 还是 self.validation_inputs 中获取数据。.iloc[index] 用于选择指定索引的行
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]

		fields_ls = self.input_fields
		# 对于缺省项的处理
		# 对于每个字段，使用 str(inputs[field])[:taken_ls[i]] 截取指定长度的字符。如果截取的字符串长度不足 taken_ls[i] 指定的长度，则通过 while len(entry) < taken_ls[i]: entry += '_' 添加下划线直到达到所需长度
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])[:taken_ls[i]]
			while len(entry) < taken_ls[i]:
				entry += '_'  # 填补符
			string_arr.append(entry)

		string = ''.join(string_arr)  # 这里应该加一个分隔符
		return string


	@classmethod
	# 此函数旨在将一个输入字符串转换为一个 PyTorch 张量，并且最终输出是一维
	# 相当于对每个字符执行 one-hot 编码
	# 具体解释如下：
	# 1 创建字符到整数的映射 (places_dict)： 首先，它创建一个字典 places_dict，将数字字符 '0' 到 '9' 映射到相应的整数值 0 到 9，然后将四个特殊字符 '. -:_' 映射到从 10 开始的整数值，每个字符对应的整数值依次增加。
	# 2 定义张量的形状 (tensor_shape) 并初始化张量 (tensor)： 根据输入字符串 input_string 的长度，定义一个三维张量的形状，第一维对应输入字符串的长度，第二维固定为 1，第三维为 15（可能是考虑到 places_dict 中有 14 个不同的字符映射）。然后，使用 torch.zeros 创建一个相应形状的全零张量。
	# 3 填充张量： 遍历输入字符串 input_string，对于字符串中的每个字符，使用 places_dict 查找该字符对应的整数值。这个整数值然后用作第三维的索引，将张量在对应位置上的值设置为 1.0。这相当于对每个字符执行 one-hot 编码。
	# 4 扁平化张量： 通过调用 tensor.flatten()，将张量从三维降为一维。这步骤通常是为了将编码后的数据作为神经网络模型的输入，因为很多模型预期输入是一维向量。
	# 5 返回张量： 最后，函数返回这个扁平化后的张量。
	# 整个过程可以被视为一个将字符串转换为机器学习模型可以处理的形式的预处理步骤。通过这种方式，每个字符都被编码为一个稀疏的 one-hot 向量，所有这些向量串联在一起，然后被扁平化成一个一维张量，这样可以直接用作模型的输入。
	def string_to_tensor(self, input_string):
		"""
		Convert a string into a tensor

		Args:
			string: str, input as a string

		Returns:
			tensor: torch.Tensor() object
		"""
		# 这行代码是一个简洁的Python字典推导式，其作用是创建一个字典，其中的键和值都来自于字符串 '0123456789'。字符串 '0123456789' 包含了从 0 到 9 的所有数字字符
		places_dict = {s:int(s) for s in '0123456789'}# int(s)：这会将字符 s 转换成一个整数。例如，如果 s 是字符 '2'，int(s) 就会是整数 2
		for i, char in enumerate('. -:_|'):# enumerate('. -:_')：这个函数遍历字符串 '. -:_'，该字符串包含四个特殊字符。enumerate 函数会为每个字符生成一个元组，其中包含一个自动递增的索引（从 0 开始）和对应的字符。例如，对于这个字符串，enumerate 会生成 (0, '.'), (1, ' '), (2, '-'), (3, ':'), (4, '_') 这样的序列。
			places_dict[char] = i + 10# places_dict[char] = i + 10：这行代码将每个特殊字符 char 作为键，将其对应的索引值 i 加上 10 作为值，存入 places_dict 字典中。这样做是为了给这些特殊字符指定一个唯一的整数值，且这些值不与前面已经指定给数字字符的值冲突。

		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(input_string), 1, 16) # 定义一个张量
		tensor = torch.zeros(tensor_shape)
		# 这段代码进一步操作了之前通过 torch.zeros(tensor_shape) 创建的三维全零张量 tensor
		for i, letter in enumerate(input_string):
			# 这行代码设置张量的一个特定位置为 1.0。这里的位置由三个索引确定：第一个索引 i 对应于 input_string 中字符的位置；第二个索引是固定的 0，因为在之前创建张量时第二维的大小被设定为 1；第三个索引是 places_dict[letter]，即当前字符在 places_dict 中映射的整数。
			# 这个过程实际上是在为 input_string 中的每个字符在张量中标记一个位置，通过将相应位置的值从 0 改为 1 来实现。
			tensor[i][0][places_dict[letter]] = 1.# places_dict[letter] 使用 letter 作为键从字典 places_dict 中获取对应的值。

		tensor = tensor.flatten()
		return tensor

	# 这个函数没有被用到
	def random_sample(self):
		"""
		Choose a random index from a training set

		Args:
			None

		Returns:
			output: string
			input: string
			output_tensor: torch.Tensor
			input_tensor: torch.Tensor
		"""
		index = random.randint(0, len(self.training_inputs['store_id']) - 1)

		output = self.training_outputs['etime'][index]
		output_tensor = torch.tensor(output)

		input_string = self.stringify_inputs(index)
		input_tensor = self.string_to_tensor(input_string)

		return output, input, output_tensor, input_tensor


	def sequential_tensors(self, training=True):
		"""
		kwargs:
			training: bool

		Returns:
			input_tensors: torch.Tensor objects
			output_tensors: torch.Tensor objects
		"""

		input_tensors = []
		output_tensors = []
		if training:
			inputs = self.training_inputs
			outputs = self.training_outputs
		else:
			inputs = self.validation_inputs
			outputs = self.validation_outputs

		for i in range(len(inputs)):
			input_string = self.stringify_input(i, training=training)
			input_tensor = self.string_to_tensor(input_string)
			input_tensors.append(input_tensor)

			# convert output float to tensor directly
			output_tensors.append(torch.Tensor([outputs[i]]))

		return input_tensors, output_tensors


class ActivateNet:

    def __init__(self, prediction_feature, epochs, ints_only=True):
        # prediction_feature是要预测的特征名，epochs是训练周期的次数，ints_only是一个布尔值，用于指定是否仅使用整数和一些特殊字符进行训练

        # specify training and test data
        # 指定了训练和测试数据文件的路径
        file = 'data/linear_historical.csv'
        # 检查ints_only参数是否为真。如果为真，表示数据预处理时仅考虑整数和特定字符
        if ints_only:
            #  计算并存储特定字符集的长度，这里包括了十个数字、小数点、空格、减号、冒号和下划线，共15个可能的字符
            n_letters = len('0123456789. -:_')  # 15 possible characters，列举了十五个可能的字符
            # 创建一个Format实例用于数据的格式化处理，传入文件路径、预测特征和训练标志
            form = Format(file, training=True)
            # 使用Format实例的sequential_tensors方法获取训练数据的输入和输出张量
            self.input_tensors, self.output_tensors = form.sequential_tensors(training=True)
            # 使用相同的方法获取验证数据的输入和输出张量
            self.validation_inputs, self.validation_outputs = form.sequential_tensors(training=False)
        # 如果ints_only为假，表示使用所有可打印字符进行数据处理
        else:
            print("wrong")
            #  获取所有可打印字符的数量
            #self.validation_inputs, self.validation_outputs = form.transform_to_tensors(training=False)

        self.epochs = epochs

        # 设置模型输出大小为1，意味着模型预测一个值
        output_size = 1
        # 计算输入张量的大小，即特征数量
        input_size = len(self.input_tensors[0])
        hidden_sizes = [512, 512, 256, 256, 128, 128]
        # 创建一个多层感知机模型实例，并将其移动到指定的计算设备上（CPU或GPU）
        self.model = DeepResidualMLP(input_size, output_size, hidden_sizes, return_embeddings=True).to(device)
        # 初始化Adam优化器，用于模型的参数优化，学习率设置为0.0001
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.biases_arr = [[], [], []]  # for plotting bias 初始化一个用于绘制偏置的空列表
        self.embedding_dim = n_letters  # 存储用于模型嵌入层的维度大小，即字符集的长度

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
        print(table)
        # 打印出可训练参数的总数量
        print(f'Total trainable parameters: {total_params}')
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
            loss = torch.mean((2 * (output - target)) ** 2)
        # 如果预测值没有低估真实值（即预测值大于或等于真实值）
        else:
            # 计算标准的均方误差损失，通过求预测值与真实值差的平方的平均数。
            loss = torch.mean((output - target) ** 2)

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
        output, *_ = self.model(input_tensor)
        # 将output_tensor（目标张量）重塑为与小批量大小相匹配的形状，每个小批量有1个输出。
        output_tensor = output_tensor.reshape(minibatch_size, 1)
        # 创建一个L1损失函数实例，用于计算预测值与实际值之间的差的绝对值的平均。
        loss_function = torch.nn.L1Loss()
        # 使用L1损失函数计算模型输出和目标张量之间的损失。
        # print(output.shape)
        # print(output_tensor.shape)
        loss = loss_function(output, output_tensor)
        # 清除（重置）模型参数的梯度，为新的梯度计算做准备。这是避免在多个小批量之间梯度累加的重要步骤。
        self.optimizer.zero_grad()  # prevents gradients from adding between minibatches
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
        self.model.eval()  # switch to evaluation mode (silence dropouts etc.)
        # 初始化三个列表model_outputs, targets, origin_ids，分别用于存储模型的预测值、真实目标值和原始数据的索引
        model_outputs, targets, origin_ids = [], [], []
        # 在这个代码块内部，不计算梯度，这样可以加速预测过程并减少内存使用
        with torch.no_grad():
            total_error = 0
            for i in range(len(self.validation_inputs)):
                # 将输入和输出张量移动到指定的计算设备上
                input_tensor = self.validation_inputs[i].unsqueeze(0).to(device)
                output_tensor = self.validation_outputs[i].to(device)

                model_output, *_ = self.model(input_tensor)
                model_outputs.append(float(model_output))
                origin_ids.append(i)
                targets.append(float(output_tensor))

        # 将模型的预测输出和真实输出值转换为浮点数，然后分别添加到model_outputs和targets列表中
        # 使用matplotlib.pyplot（别名plt）和plotly.express（别名px）绘制散点图，将真实值作为x轴，
        # 预测值作为y轴，通过不同的颜色表示不同的数据点
        plt.scatter(targets, model_outputs, s=1.5)
        fig = px.scatter(x=targets,
                         y=model_outputs)
        # color=origin_ids,
        # color_continuous_scale=["blue","blue","blue","blue","blue","blue","blue"])

        fig.update_traces(textposition='top center')
        # fig.show()

        # _, _, rval, _, _ = scipy.stats.linregress([float(i) for i in self.validation_outputs], model_outputs)
        # print (f'R2 value: {rval**2}')
        # plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
        # plt.show()
        # plt.savefig('filename.png')
        plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签的字体大小
        plt.rcParams['ytick.labelsize'] = 14
        # 调整图形布局，使之紧凑
        plt.axis([-100, 10000, -100, 8000])
        plt.xticks(np.arange(0, 10000, 2000))
        plt.yticks(np.arange(0, 8000, 2000))
        plt.tight_layout()
        # 保存图形为PNG格式，文件名包含训练的epoch编号
        plt.savefig('regression_residual/DeepResidualMLP_regression{0:04d}_fcnet.png'.format(epoch_number), dpi=400)
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
        plt.plot([i[0] for i in self.biases_arr[0]], [i[1] for i in self.biases_arr[0]], '^', color='white', alpha=0.7,
                 markersize=0.1)
        plt.plot([i[0] for i in self.biases_arr[1]], [i[1] for i in self.biases_arr[1]], '^', color='red', alpha=0.7,
                 markersize=0.1)
        plt.plot([i[0] for i in self.biases_arr[2]], [i[1] for i in self.biases_arr[2]], '^', color='blue', alpha=0.7,
                 markersize=0.1)
        # 确保坐标轴是开启状态，以便在图像中显示
        plt.axis('on')
        # 保存绘制的图像到文件系统，文件名中包含传入的index参数，以便于跟踪和比较不同时间点的偏置变化
        plt.savefig('DeepResidualMLP_{0:04d}.png'.format(index), dpi=400)
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
        arr = arr[:44 * 44]
        # 将这1936个元素重塑为44行44列的二维张量，准备以热图形式展示
        arr = torch.reshape(arr, (44, 44))
        # 将PyTorch张量转换为NumPy数组，以便使用matplotlib进行绘图
        arr = arr.numpy()
        # 使用matplotlib的imshow函数绘制热图，interpolation='spline16'设置插值方式，aspect='auto'自动调整长宽比，cmap='inferno'设置颜色映射。
        plt.imshow(arr, interpolation='spline16', aspect='auto', cmap='inferno')
        plt.style.use('dark_background')
        plt.axis('off')
        # 保存热图到文件，文件名包含传入的index参数，dpi=400设置图像分辨率，bbox_inches='tight', pad_inches=0去除周围空白边距
        plt.savefig('DeepResidualMLP_heatmap_weight_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight',
                    pad_inches=0)
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
        plt.savefig('DeepResidualMLP_heatmap_biases_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight',
                    pad_inches=0)
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
            self.model.eval()  # switch to evaluation mode (silence dropouts etc.)
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
                for j in range(i + 1, len(embedding_arr)):
                    actual_distances.append(np.abs(actual_arr[j] - actual_arr[i]))
                    embedding_distance = torch.sum(
                        torch.abs(embedding_arr[j] - embedding_arr[i])).cpu().detach().numpy()
                    input_distance = torch.sum(
                        torch.abs(input_arr[j][14 * 15:31 * 15] - input_arr[i][14 * 15:31 * 15])).cpu().detach().numpy()
                    embedding_distances.append(embedding_distance)
                    input_distances.append(input_distance)
                    origin_id.append(i)
                    target_id.append(j)

            plt.rcParams.update({'font.size': 17})
            plt.scatter(actual_distances, embedding_distances, s=0.3)
            # plt.scatter(actual_distances, embedding_distances, s=0.3)
            plt.xlabel('Actual Distance')
            plt.ylabel('Embedding Distance')
            plt.tight_layout()
            plt.savefig('DeepResidualMLP_embedding/DeepResidualMLP_embedding{0:04d}.png'.format(index), dpi=350)
            plt.close()

            fig = px.scatter(x=actual_distances,
                             y=embedding_distances,
                             )
            fig.update_traces(textposition='top center')

            plt.scatter(input_distances, embedding_distances, s=0.3)
            plt.xlabel('Input Distance')
            plt.ylabel('Embedding Distance')
            plt.rcParams.update({'font.size': 18})
            plt.savefig(
                'DeepResidualMLP_input_embedding_residual/DeepResidualMLP_input_embedding_residual.png'.format(index),
                dpi=390)
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
                """
                if count % 100 == 0:
                    self.plot_embedding(index=count // 100)
                    # interpret = StaticInterpret(self.model, self.validation_inputs, self.validation_outputs,self.embedding_dim)
                    self.plot_predictions(count // 25)  # count//100
                """
                # if count % 100 == 0:
                # 	self.plot_embedding(index=count//100)
                #   interpret = StaticInterpret(self.model, self.validation_inputs, self.validation_outputs)
                #   self.plot_predictions(count//25)
                #   interpret.heatmap(count, method='combined')
                count += 1

            print(f'Epoch {epoch} complete: {total_loss} loss')
        plt.figure(figsize=(10, 5))
        plt.plot(losses, label='Loss per Batch')
        plt.xlabel('Batch Number')
        plt.ylabel('Loss')
        plt.title('Loss per Batch During Training')
        plt.savefig('loss_residual/loss_residual.png', dpi=400)
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
        self.model.eval()  # switch to evaluation mode (silence dropouts etc.)
        loss = torch.nn.L1Loss()
        # print("1")
        total_error = 0
        # print("2")
        for i in range(len(self.validation_inputs)):
            input_tensor = self.validation_inputs[i].unsqueeze(0).to(device)
            output_tensor = self.validation_outputs[i].to(device)
            # print("3")
            # print(input_tensor.shape)
            model_output, *_ = self.model(input_tensor)
            # print("4")
            total_error += loss(model_output, output_tensor).item()
            # print("5")
        print(f'Test Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')

        total_error = 0
        for i in range(len(self.input_tensors[:n_taken])):
            input_tensor = self.input_tensors[i].unsqueeze(0)
            output_tensor = self.output_tensors[i]
            model_output, *_ = self.model(input_tensor.to(device))
            total_error += loss(model_output, output_tensor.to(device)).item()

        print(f'Training Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')
        return


if __name__ == '__main__':
    network = ActivateNet('positive_control', 200)  # 800
    network.train_model()
    # hidden
    network.test_model()  # 这个跑不起来
"""
    torch.save(network.model.state_dict(), 'EMLP_RAM_model_weights')
    # network.model.load_state_dict(torch.load('model_weights'))
    # network.heatmap_biases(0)
    network.plot_embedding() # 这个也是
    network.plot_predictions(0)# 这个也是
    # 下面是自己加的
    torch.save(network.model.state_dict(), 'model_weights')
    network.model.load_state_dict(torch.load('model_weights'))
    network.test_model()

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
"""





