# 定义并实现了用于处理分类输出的多层感知器(MLP)模型
# 包括数据预处理、模型训练、评估和预测的功能。
# 该代码概述了具有多个隐藏层的神经网络的结构，并采用了ReLU激活和dropout等技术来防止过拟合。
# 此外，它还引入了自定义数据格式化和加载类来处理输入数据，包括将字符串数据转换为适合模型输入的张量。
# 该网络使用专门的损失函数进行训练，可能是为了解决特定的问题特征，如不平衡数据或成本敏感学习。
# 代码还包含用于评估模型在验证集上的性能和可视化预测、偏差和梯度的方法，以解释模型的行为。
#
# fcnet_categorical.py
# MLP-style model for categorical outputs

# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import sklearn
from sklearn.utils import shuffle
import scipy

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from network_interpret import CategoricalStaticInterpret as interpret
from data_formatter import Format as GeneralFormat
from network_interpret import StaticInterpret
# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)

# 多层过程
class MultiLayerPerceptron(nn.Module):

	def __init__(self, input_size, output_size):

		super().__init__()
		self.input_size = input_size
		hidden1_size = 500
		hidden2_size = 100
		hidden3_size = 20
		self.input2hidden = nn.Linear(input_size, hidden1_size)
		self.hidden2hidden = nn.Linear(hidden1_size, hidden2_size)
		self.hidden2hidden2 = nn.Linear(hidden2_size, hidden3_size)
		self.hidden2output = nn.Linear(hidden3_size, output_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.)

	# 定义了一个神经网络的前向传播过程
	def forward(self, input):
		"""
		Forward pass through network

		Args:
			input: torch.Tensor object of network input, size [n_letters * length]

		Return: 
			output: torch.Tensor object of size output_size

		"""
		# 这一行将输入数据input（一个torch.Tensor对象，大小为[n_letters * length]）通过一个名为input2hidden的层进行处理。
		# 这个层可能是一个全连接层（Linear层），它会根据输入的尺寸和该层定义的输出尺寸进行线性变换。
		out = self.input2hidden(input)
		# 应用ReLU激活函数到上一步的输出上。
		# ReLU（Rectified Linear Unit）激活函数的作用是将所有负值置为0，保留所有正值，这有助于增加模型的非线性并解决梯度消失的问题。
		out = self.relu(out)
		# 应用dropout操作到上一步的输出上。
		# Dropout是一种正则化技术，通过在训练过程中随机“丢弃”一部分神经元的输出（将它们设为0），以此来防止模型过拟合。
		# dropout函数的参数控制丢弃神经元的比例。
		out = self.dropout(out)
		# 以下两个段落重复了第1到第3步，但是使用了不同的层hidden2hidden和hidden2hidden2进行处理。
		# 这表明网络中有多个隐藏层，每个隐藏层后都跟着ReLU激活和dropout正则化
		out = self.hidden2hidden(out)
		out = self.relu(out)
		out = self.dropout(out)

		out = self.hidden2hidden2(out)
		out = self.relu(out)
		out = self.dropout(out)
		# 将最后一个隐藏层的输出通过另一个全连接层（hidden2output）进行变换，以产生最终的输出output。
		# 这个输出的大小应该与模型需要预测的类别数量相匹配。
		output = self.hidden2output(out)
		return output


class Format():

	def __init__(self, file, training=True, n_per_field=False, deliveries=False):

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower() == 'nan' else x)
		df = df[:100000]
		length = len(df[:])
		self.input_fields = ['PassengerId',
							 'Pclass',
							 'Name',
							 'Sex',
							 'Age',
							 'SibSp',
							 'Parch',
							 'Ticket',
							 'Fare',
							 'Cabin',
							 'Embarked','H']
		if n_per_field:
			self.taken_ls = [4 for i in self.input_fields]
		else:
			self.taken_ls = [3, 1, 5, 2, 3, 2, 4, 5, 4, 4, 1]

		if training:
			df = shuffle(df)
			df.reset_index(inplace=True)

			# 80/20 training/test split
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[self.input_fields]
			self.training_outputs = [i for i in training['Survived'][:]]

			df2 = pd.read_csv('titanic/test.csv')
			validation_size = len(df2)
			validation = df2
			self.validation_inputs = validation[self.input_fields]
			df3 = pd.read_csv('titanic/gender_submission.csv')
			self.validation_outputs = [i for i in df3['Survived'][:]] 
			self.validation_inputs = self.validation_inputs.reset_index()

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify

	# 将数据帧中的特定行转换为字符串，方法是将每个字段的值转换为字符串并连接起来。
	# 如果pad=True，则会根据length参数在必要时填充或截断字符串，以确保字符串具有一致的长度。
	def unstructured_stringify(self, index, training=True, pad=True, length=75):
		"""
		Compose array of string versions of relevant information in self.df 
		Does not maintain a consistant structure to inputs regardless of missing 
		values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""
		# string_arr = []初始化一个空列表，用于存储转换后的字符串值
		string_arr = []
		# 根据training的值，从self.training_inputs或self.validation_inputs属性中选择相应的数据行。
		# .iloc[index]通过位置索引选择数据，确保从正确的数据集中获取指定行的数据。
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]
		# 获取模型输入字段的列表，这些字段定义了哪些列将被包含在最终的字符串中。
		fields_ls = self.input_fields
		# 遍历输入字段列表，enumerate函数同时返回字段的索引（i）和名称（field）
		for i, field in enumerate(fields_ls):
			# 从当前行中获取指定字段的值，并将其转换为字符串。这保证了无论原始数据类型如何，结果都是字符串格式。
			entry = str(inputs[field])
			# 将转换后的字段值添加到之前初始化的字符串数组中
			string_arr.append(entry)

		# 使用空字符串作为连接符，将所有字段值连接成一个单一的字符串。
		string = ''.join(string_arr)
		# 如果pad为True，则根据字符串的实际长度与目标长度length的比较结果，对字符串进行填充或截断：
		# 如果字符串长度小于目标长度，则在字符串末尾添加足够数量的下划线（'_'），使其长度达到length。
		# 如果字符串长度大于目标长度，则将字符串截断到length指定的长度。
		if pad:
			if len(string) < length:
				string += '_' * (length - len(string))
			if len(string) > length:
				string = string[:length]
		return string

	# 将数据集中指定行的数据转换成一个固定格式的字符串
	# 数定义：def stringify_input(self, index, training=True):定义了一个名为stringify_input的方法，它接受两个参数：
	# self：类的实例自身。
	# index：整数，表示要处理的数据行的位置。
	# training：布尔值，默认为True，表示是否从训练集中选择数据。如果为False，则从验证集中选择数据。
	def stringify_input(self, index, training=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""
		# 从类的实例中获取taken_ls列表，该列表包含每个字段应取的字符数
		taken_ls = self.taken_ls
		# 初始化一个空列表，用于存储转换后的各字段字符串值
		string_arr = []
		# 根据training的值，从self.training_inputs或self.validation_inputs属性中选择相应的数据行。
		# .iloc[index]通过位置索引选择数据，确保从正确的数据集中获取指定行的数据
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]
		# fields_ls = self.input_fields从类实例中获取模型输入字段的列表，这些字段定义了哪些列将被包含在最终的字符串中
		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])[:taken_ls[i]]
			while len(entry) < taken_ls[i]:
				entry += '_'
			string_arr.append(entry)
		# string = ''.join(string_arr)使用''.join方法将列表中的所有字符串值连接成一个单一的字符串
		string = ''.join(string_arr)
		return string

	# 将一个字符串转换为张量（tensor），适用于将文本或数字输入转换为神经网络可以处理的格式
	@classmethod
	def string_to_tensor(self, input_string, ints_only=False):
		# ints_only：布尔值，默认为False。如果为True，则只考虑数字和一些特殊字符（如小数点、负号等）的转换；否则，会考虑所有可打印字符的转换
		"""
		Convert a string into a tensor. For numerical inputs.

		Args:
			string: str, input as a string

		Returns:
			tensor
		"""
		# 条件创建字典
		# 如果ints_only为True，则创建一个places_dict字典，其中包含数字和一些特殊字符（'0123456789. -:_'），每个字符映射到一个唯一的整数索引。这用于处理仅包含数字的输入。
		# 否则，使用string.printable获取所有可打印的字符，并为这些字符创建同样的映射。string.printable包含数字、英文字母、标点符号以及空格。
		if ints_only:
			places_dict = {s:i for i, s in enumerate('0123456789. -:_')}

		else:
			chars = string.printable
			places_dict = {s:i for i, s in enumerate(chars)}
		# 设置嵌入维度为places_dict字典的长度，即考虑的字符集大小
		self.embedding_dim = len(places_dict)
		# vocab_size x batch_size x embedding dimension (ie input length)
		# 定义要创建的张量的形状。这里，张量的形状取决于输入字符串的长度、批处理大小（这里为1，因为处理单个字符串），以及嵌入维度（即字符集大小）
		tensor_shape = (len(input_string), 1, len(places_dict))
		# 创建一个给定形状的全零张量，作为字符到张量的转换的基础
		tensor = torch.zeros(tensor_shape)
		# 遍历输入字符串中的每个字符，使用字符到索引的映射（places_dict）找到每个字符对应的索引，并在张量的相应位置上设置为1，实现独热编码。
		# 这意味着每个字符都被转换为一个只在对应字符索引处为1，其他位置为0的向量
		for i, letter in enumerate(input_string):
			tensor[i][0][places_dict[letter]] = 1.
		# 将张量扁平化为一维，以便可以作为神经网络的输入。这是因为网络期望接收的输入通常是一维向量
		tensor = tensor.flatten()
		return tensor 


	# 将输入数据（可能是文本或其他形式）和输出数据转换为神经网络可以处理的张量形式
	def sequential_tensors(self, training=True):
		"""
		
		"""
		# 初始化两个空列表，用于存储转换后的输入和输出张量
		input_tensors = []
		output_tensors = []
		# 条件选择数据集
		if training:
			inputs = self.training_inputs
			outputs = self.training_outputs
		else:
			inputs = self.validation_inputs
			outputs = self.validation_outputs

		# 循环处理每个输入
		# len(inputs)给出了数据集中元素的数量，range生成一个序列，用于索引数据集中的每个元素
		for i in range(len(inputs)):
			#
			#输入转换为字符串
			#
			# 调用stringify_input方法将索引为i的输入数据转换为字符串。
			# 这个方法可能会根据数据的特点（如文本数据）进行适当的格式化和处理。
			# training=training确保了处理数据时使用正确的数据集（训练或验证）
			input_string = self.stringify_input(i, training=training)
			#
			# 字符串转换为张量
			#
			# 使用string_to_tensor方法将上一步得到的字符串转换为张量。这个过程涉及到字符的编码和可能的维度转换，以生成适合神经网络处理的张量
			input_tensor = self.string_to_tensor(input_string)
			#
			# 存储输入张量
			#
			# 将转换后的输入张量添加到input_tensors列表中
			input_tensors.append(input_tensor)

			# convert output float to tensor directly
			# 直接将输出数据转换为张量并添加到output_tensors列表中。
			# 这里使用了torch.tensor构造函数，[outputs[i]]将单个输出值包装在列表中以创建一个张量。
			# 这通常用于处理数值型输出，如回归任务中的连续值
			output_tensors.append(torch.tensor([outputs[i]]))

		return input_tensors, output_tensors


# ActivateNet类主要负责神经网络的初始化、训练、评估和预测。
# 它支持对特定数据集（如示例中的Titanic数据集）进行处理和模型训练。
# 根据是否指定deliveries参数，ActivateNet会以不同方式初始化数据集，进而构建输入和输出张量。
# 此类还实现了自定义的损失函数（weighted_mseloss和weighted_l1loss），这些函数可以根据特定的业务需求定制损失计算方式。
# 此外，ActivateNet提供了train_minibatch、train_model、train_online、test_model等方法用于训练和测试模型，
# 并包含了辅助方法如plot_predictions、plot_biases和quiver_gradients等用于可视化模型性能和内部参数
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

		print (len(self.input_tensors), len(self.validation_inputs))
		self.epochs = epochs
		output_size = 2
		input_size = len(self.input_tensors[0])
		self.model = MultiLayerPerceptron(input_size, output_size)
		self.model.to(device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.biases_arr = [[], []]

	# 定义了一个定制的均方误差（MSE）损失函数，用于考虑预测误差的不同权重
	# 这个函数通过对低估情况加重惩罚，定制了传统的MSE损失函数，以适应特定的应用场景，其中低估的代价比高估的代价要大
	def weighted_mseloss(self, output, target):
		# 它接受两个参数output和target，分别是模型的预测输出和真实目标值，都是torch.tensor类型

		"""
		We are told that the true cost of underestimation is twice
		that of overestimation, so MSEloss is customized accordingly.

		Args:
			output: torch.tensor
			target: torch.tensor

		Returns:
			loss: float

		"""
		# 计算加权损失（低估情况）：loss = torch.mean((2*(output - target))**2)。
		# 在低估情况下，损失计算会给误差乘以2，再求平方，表示低估的真实成本是高估的两倍。
		# 然后使用torch.mean计算这个加权误差平方的均值，得到最终的损失值
		if output < target:
			loss = torch.mean((2*(output - target))**2)
		# 计算标准MSE损失（高估情况）：else: loss = torch.mean((output - target)**2)。
		# 如果预测值不小于目标值（即模型高估或精确预测了真实值），则直接计算预测值和目标值之差的平方的均值，作为损失值。
		else:
			loss = torch.mean((output - target)**2)

		return loss

	# 定义了一个定制的L1损失（绝对值损失），其中对于预测值低于真实值的情况赋予了双重权重
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

	# 训练神经网络的一个小批量（minibatch）数据，并返回该批次的模型预测和损失值
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
		# 将输入张量送入模型进行前向传播，得到预测输出。input_tensor.to(device)确保输入张量被移至正确的设备上（比如GPU）
		output = self.model(input_tensor.to(device))
		# 将目标张量的形状调整为与小批量大小一致，并确保其被移至正确的设备
		output_tensor = output_tensor.reshape(minibatch_size).to(device)
		# 将目标张量的数据类型转换为长整型，这通常是分类任务中的要求
		output_tensor = output_tensor.long()
		# loss_function = torch.nn.L1Loss()
		# 使用交叉熵损失函数，这是分类任务中常用的损失函数
		loss_function = torch.nn.CrossEntropyLoss()
		# 根据模型的预测输出和真实目标计算损失值
		loss = loss_function(output, output_tensor)
		# 在每次的反向传播之前清零累积的梯度，防止在小批量之间的梯度相互影响
		self.optimizer.zero_grad() # prevents gradients from adding between minibatches
		# 根据损失函数计算梯度，并通过反向传播算法将其反向传播回网络的参数
		loss.backward()
		# 根据计算得到的梯度更新模型的参数
		self.optimizer.step()
		# 返回模型的预测输出output和该小批量的损失值loss.item()（通过调用.item()方法将单元素张量的值转换为Python标量）
		return output, loss.item()

	# 在模型的评估模式下，对验证集的预测结果进行绘图，并保存为图片文件
	def plot_predictions(self, epoch_number):
		"""

		"""
		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()
		model_outputs = []

		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i].to(device)
				output_tensor = self.validation_outputs[i].to(device)
				model_output = self.model(input_tensor)
				# 原本：model_outputs.append(float(model_output))
				# 修改：
				if model_output.numel() == 1:
					model_outputs.append(model_output.item())
				else:
					# 可选：处理多元素输出
					model_outputs.append(model_output.mean().item())


		plt.scatter([float(i) for i in self.validation_outputs], model_outputs, s=15)
		plt.axis([0, 1600, -100, 1600]) # x-axis range followed by y-axis range
		plt.show()
		plt.tight_layout()
		plt.savefig('regression_fcnet_cata/regression_fc_cata{0:04d}.png'.format(epoch_number), dpi=400)
		plt.close()
		return

	def plot_biases(self, index):
		"""

		"""
		x, y = self.model.hidden2hidden2[:2].detach().numpy()
		self.biases_arr[0].append(x)
		self.biases_arr[1].append(y)
		plt.style.use('dark_background')
		plt.plot(x_arr, y_arr, '^', color='white', alpha=2, markersize=0.1)
		plt.axis('on')
		plt.savefig('Biases_{0:04d}.png'.format(index), dpi=400)
		plt.close()
		return

	# 绘制模型特定层偏置的梯度场
	def quiver_gradients(self, index, input_tensor, output_tensor, minibatch_size=64):
		"""
		plots

		"""
		self.model.eval()
		x, y = self.model.hidden2hidden.bias[:2].detach().numpy()
		print (x, y)
		plt.style.use('dark_background')

		x_arr = np.arange(x - 0.01, x + 0.01, 0.001)
		y_arr = np.arange(y - 0.01, y + 0.01, 0.001)

		XX, YY = np.meshgrid(x_arr, y_arr)
		dx, dy = np.meshgrid(x_arr, y_arr) # copy that will be overwritten
		for i in range(len(x_arr)):
			for j in range(len(y_arr)):
				with torch.no_grad():
					self.model.hidden2hidden.bias[0] = torch.nn.Parameter(torch.Tensor([x_arr[i]]))
					self.model.hidden2hidden.bias[1] = torch.nn.Parameter(torch.Tensor([y_arr[j]]))
				output = self.model(input_tensor)
				output_tensor = output_tensor.reshape(minibatch_size, 1)
				loss_function = torch.nn.L1Loss()
				loss = loss_function(output, output_tensor)
				self.optimizer.zero_grad()
				loss.backward()
				dx[j][i], dy[j][i] = self.model.hidden2hidden.bias.grad[:2]

		matplotlib.rcParams.update({'font.size': 8})
		color_array = 2*(np.abs(dx) + np.abs(dy))
		# 使用plt.quiver绘制梯度矢量场，color_array基于梯度大小设置颜色强度
		plt.quiver(XX, YY, dx, dy, color_array)
		plt.plot(x, y, 'o', markersize=1)
		plt.savefig('quiver_{0:04d}.png'.format(index), dpi=400)
		plt.close()
		with torch.no_grad():
			self.model.hidden2hidden.bias[:2] = torch.Tensor([x, y])
		return

	# 函数旨在对两个隐藏层的偏置梯度进行可视化，通过生成两个梯度场图来比较它们
	def quiver_gradients_double(self, index, input_tensor, output_tensor, minibatch_size=64):
		"""

		"""
		self.model.eval()
		x, y = self.model.hidden2hidden.bias[:2].detach().numpy()
		x_arr = np.arange(x - 0.1, x + 0.1, 0.02)
		y_arr = np.arange(y - 0.1, y + 0.1, 0.01)

		XX, YY = np.meshgrid(x_arr, y_arr)
		dx, dy = np.meshgrid(x_arr, y_arr) # copy that will be overwritten
		for i in range(len(x_arr)):
			for j in range(len(y_arr)):
				with torch.no_grad():
					self.model.hidden2hidden.bias[0] = torch.nn.Parameter(torch.Tensor([x_arr[i]]))
					self.model.hidden2hidden.bias[1] = torch.nn.Parameter(torch.Tensor([y_arr[j]]))
				output = self.model(input_tensor)
				output_tensor = output_tensor.reshape(minibatch_size, 1)
				loss_function = torch.nn.L1Loss()
				loss = loss_function(output, output_tensor)
				self.optimizer.zero_grad()
				loss.backward()
				dx[j][i], dy[j][i] = self.model.hidden2hidden.bias.grad[:2]

		x2, y2 = self.model.hidden2hidden2.bias[:2].detach().numpy()

		x_arr2 = np.arange(x2 - 0.1, x2 + 0.1, 0.02)
		y_arr2 = np.arange(y2 - 0.1, y2 + 0.1, 0.01)

		XX2, YY2 = np.meshgrid(x_arr2, y_arr2)
		dx2, dy2 = np.meshgrid(x_arr2, y_arr2) # copy that will be overwritten
		for i in range(len(x_arr2)):
			for j in range(len(y_arr2)):
				with torch.no_grad():
					self.model.hidden2hidden2.bias[0] = torch.nn.Parameter(torch.Tensor([x_arr2[i]]))
					self.model.hidden2hidden2.bias[1] = torch.nn.Parameter(torch.Tensor([y_arr2[j]]))
				output = self.model(input_tensor)
				output_tensor = output_tensor.reshape(minibatch_size, 1)
				loss_function = torch.nn.L1Loss()
				loss = loss_function(output, output_tensor)
				self.optimizer.zero_grad()
				loss.backward()
				dx2[j][i], dy2[j][i] = self.model.hidden2hidden2.bias.grad[:2]

		
		color_array = 2*(np.abs(dx) + np.abs(dy))
		matplotlib.rcParams.update({'font.size': 7})
		plt.style.use('dark_background')
		plt.subplot(1, 2, 1)
		plt.quiver(XX, YY, dx, dy, color_array)
		plt.title('Hidden Layer 1')

		plt.subplot(1, 2, 2)
		color_array2 = 2*(np.abs(dx2) + np.abs(dy2))
		plt.quiver(XX2, YY2, dx2, dy2, color_array2)
		plt.title('Hidden Layer 2')
		plt.savefig('quiver_{0:04d}.png'.format(index), dpi=400)
		plt.close()

		with torch.no_grad():
			self.model.hidden2hidden.bias[:2] = torch.Tensor([x, y])
			self.model.hidden2hidden2.bias[:2] = torch.Tensor([x2, y2])
		return

	# 用于训练多层感知机模型的。函数按照指定的批次大小(minibatch_size)和预设的训练周期(epochs)进行迭代训练
	def train_model(self, minibatch_size=128):
		"""
	    Train the mlp model

	    Args:
	          model: MultiLayerPerceptron object
	          optimizer: torch.optim object
	          minibatch_size: int
	     Returns:
	        None

	     """
		# 确保模型处于训练模式，启用Dropout等特性
		self.model.train()
		# 从类的属性中获取预定的训练周期数
		epochs = self.epochs
		# 初始化计数器：count用于记录训练过程中的迭代次数
		count = 0
		losses = []  # 初始化一个列表来保存每个批次的损失值
		# 开始训练周期迭代：通过for epoch in range(epochs):循环控制训练的周期
		for epoch in range(epochs):
			# 打印当前训练的周期数
			print(f'Epoch {epoch}')
			# 将输入张量和输出张量配对，随机打乱，以实现数据的随机访问
			pairs = [[i, j] for i, j in zip(self.input_tensors, self.output_tensors)]
			random.shuffle(pairs)
			input_tensors = [i[0] for i in pairs]
			output_tensors = [i[1] for i in pairs]
			# 初始化总损失和正确率计数：为本周期内的总损失和正确率计数做初始化
			total_loss = 0
			correct, total = 0, 0

			# 遍历批次进行训练：通过for循环以minibatch_size为步长遍历训练数据，每个批次进行一次前向传播和反向传播。
			#
			# 组装批次数据：使用torch.stack合并批次内的数据。
			#
			# 跳过小于批次大小的数据：如果剩余数据量小于一个批次大小，则跳过。
			for i in range(0, len(input_tensors) - minibatch_size, minibatch_size):
				# stack tensors to make shape (minibatch_size, input_size)
				input_batch = torch.stack(input_tensors[i:i + minibatch_size])
				output_batch = torch.stack(output_tensors[i:i + minibatch_size])

				# skip the last batch if too small
				if len(input_batch) < minibatch_size:
					break
				output, loss = self.train_minibatch(input_batch, output_batch, minibatch_size)

				total_loss += loss
				losses.append(loss)  # 保存当前批次的损失值
				output_batch = output_batch.reshape(minibatch_size).to(device)
				correct += torch.sum(torch.argmax(output, dim=1) == output_batch)
				total += minibatch_size
			# if i % 1 == 0:
			# 	print (f'Epoch {epoch} complete: {total_loss} loss')
			# 	self.quiver_gradients(count, input_batch, output_batch)
			# 	count += 1
			# self.plot_predictions(count)
			print(f'Train Accuracy: {correct / total}')
			print(f'Loss: {total_loss}')
			count += 1
		# 绘制损失值图形
		plt.figure(figsize=(10, 5))
		plt.plot(losses, label='Loss per Batch')
		plt.xlabel('Batch Number')
		plt.ylabel('Loss')
		plt.title('Loss per Batch During Training')
		plt.savefig('loss_fcnet_cata/loss_fcnet_cata.png', dpi=400)
		plt.legend()
		plt.show()
		return

	# 用于对模型进行在线训练，通过随机抽样更新梯度
	def train_online(self, file, minibatch_size=1):
		"""
		On-line training with random samples

		Args:
			model: Transformer object
			optimizer: torch.optim object of choice

		kwags:
			minibatch_size: int, number of samples per gradient update

		Return:
			none (modifies model in-place)

		"""

		self.model.train()
		current_loss = 0
		training_data = Format(file, training=True)

		# training iteration and epoch number specs
		n_epochs = 10

		start = time.time()
		for i in range(n_epochs):
			random.shuffle(input_samples)
			for i in range(0, len(self.input_samples), minibatch_size):
				if len(input_samples) - i < minibatch_size:
					break

				input_tensor = torch.cat([input_samples[i+j] for j in range(minibatch_size)])
				output_tensor = torch.cat([output_samples[i+j] for j in range(minibatch_size)])

				# define the output and backpropegate loss
				output, loss = train_random_input(output_tensor, input_tensor)

				# sum to make total loss
				current_loss += loss 

				if i % n_per_epoch == 0 and i > 0:
					etime = time.time() - start
					ave_error = round(current_loss / n_per_epoch, 2)
					print (f'Epoch {i//n_per_epoch} complete \n Average error: {ave_error} \n Elapsed time: {round(etime, 2)}s \n' + '~'*30)
					current_loss = 0 
		return


	def test_model(self):
		"""

		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()

		model_outputs, true_outputs = [], []
		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i].to(device)
				output_tensor = self.validation_outputs[i].to(device)
				model_output,*_ = self.model(input_tensor)
				total_error += loss(model_output, output_tensor).item()
				model_outputs.append(float(model_output))
				true_outputs.append(float(output_tensor))

		slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(model_outputs, true_outputs)
		print (f'Mean Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')
		print (f'R2 value: {r_value**2}')
		return

	def test_model_categories(self):
		"""

		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)

		model_outputs, true_outputs = [], []
		minibatch_size = 1
		with torch.no_grad():
			correct, count = 0, 0
			for i in range(0, len(self.validation_inputs), minibatch_size):
				input_batch = torch.stack(self.validation_inputs[i:i + minibatch_size])
				output_batch = torch.stack(self.validation_outputs[i:i + minibatch_size])
				input_tensor = input_batch.to(device)
				output_tensor = output_batch.reshape(minibatch_size).to(device)
				model_output = self.model(input_tensor)
				correct += torch.sum(torch.argmax(model_output, dim=1) == output_tensor)
				count += minibatch_size

		print (correct, count)
		print (f'Test Accuracy: {correct / count}')
		return


	def predict(self, model, test_inputs):
		"""
		Make predictions with a model.

		Args:
			model: Transformer() object
			test_inputs: torch.tensor inputs of prediction desired

		Returns:
			prediction_array: arr[int] of model predictions

		"""
		model.eval()
		prediction_array = []

		with torch.no_grad():
			for i in range(len(test_inputs['index'])):
				prediction_array.append(model_output)

		return prediction_array

epochs = 200



network = ActivateNet(epochs)
network.train_model()
network.test_model() # 用于测试回归任务
network.test_model_categories() # 用于测试分类任务

interpretation = interpret(network.model, network.validation_inputs, network.validation_outputs)
# interpretation.heatmap(0)
# interpretation.readable_interpretation(0)




