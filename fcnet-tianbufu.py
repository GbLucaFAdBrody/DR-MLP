# fcnet.py
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

# import local libraries
from network_interpret import StaticInterpret

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiLayerPerceptron(nn.Module):

	def __init__(self, input_size, output_size):

		super().__init__()
		self.input_size = input_size
		hidden1_size = 500 #900
		hidden2_size = 100
		hidden3_size = 20
		self.input2hidden = nn.Linear(input_size, hidden1_size)
		self.hidden2hidden = nn.Linear(hidden1_size, hidden2_size)
		self.hidden2hidden2 = nn.Linear(hidden2_size, hidden3_size)
		self.hidden2output = nn.Linear(hidden3_size, output_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5) #本来是0.
	# 该函数保持一致
	# 向前传播函数
	def forward(self, input):
		"""
		Forward pass through network

		Args:
			input: torch.Tensor object of network input, size [n_letters * length]

		Return: 
			output: torch.Tensor object of size output_size

		"""
		# 1
		out = self.input2hidden(input) # 得到的是一个[hidden1_size, input_size]形状的张量

		# 变换后的数据 out 被传递给 ReLU 激活函数。ReLU（Rectified Linear Unit）是一种非线性激活函数，它对输入的每个元素执行 max(0, x) 操作。
		# 这意味着所有负值都会被置为0，而所有正值保持不变。这一步是引入非线性的关键，使得神经网络可以学习并模拟非线性关系。
		out = self.relu(out)

		# 最后，ReLU 函数的输出被传递给 Dropout 层。Dropout 是一种正则化技术，用于减少神经网络在训练过程中的过拟合。
		# 它通过在训练时随机将层的部分输出单元置零（根据设定的概率 p），来减少单个神经元对于局部输入特征的依赖，从而强迫网络学习更加健壮的特征。
		# 这里没有指定 p，所以会使用 nn.Dropout 的默认值，通常是 0.5。
		out = self.dropout(out)

		# 2
		out = self.hidden2hidden(out)
		out = self.relu(out)
		out = self.dropout(out)

		# 3
		out = self.hidden2hidden2(out)
		out = self.relu(out)
		out = self.dropout(out)

		# 最后，函数返回两个值：output 和 embedding。
		# output 是网络的最终输出，可以用于计算损失、执行预测等；而 embedding 是网络中间层的输出，根据具体任务，它可能会被用于其他分析或处理步骤。
		embedding = out

		output = self.hidden2output(out)
		return output, embedding


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
			self.training_outputs = [i for i in training['positive_three'][:]]# 这行代码使用列表推导式从 training DataFrame 中提取 'positive_control' 列的所有值作为训练输出。这里 'positive_control' 应该是包含标签或目标变量的列。列表推导式 for i in training['positive_control'][:] 遍历 'positive_control' 列中的每个元素，将其添加到列表中，然后将这个列表赋值给 self.training_outputs。使用 [:] 是多余的，直接使用 training['positive_control'] 就可以达到同样的目的。

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.validation_inputs = validation[self.input_fields]
			self.validation_outputs = [i for i in validation['positive_three'][:]]
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

		string = 'K'.join(string_arr)  # 这里应该加一个分隔符
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
		for i, char in enumerate('. -:_|K'):# enumerate('. -:_')：这个函数遍历字符串 '. -:_'，该字符串包含四个特殊字符。enumerate 函数会为每个字符生成一个元组，其中包含一个自动递增的索引（从 0 开始）和对应的字符。例如，对于这个字符串，enumerate 会生成 (0, '.'), (1, ' '), (2, '-'), (3, ':'), (4, '_') 这样的序列。
			places_dict[char] = i + 10# places_dict[char] = i + 10：这行代码将每个特殊字符 char 作为键，将其对应的索引值 i 加上 10 作为值，存入 places_dict 字典中。这样做是为了给这些特殊字符指定一个唯一的整数值，且这些值不与前面已经指定给数字字符的值冲突。

		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(input_string), 1, 17) # 定义一个张量
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

	def __init__(self, epochs):
		n_letters = len('0123456789. -:_|K') # 15 possible characters
		file = 'data/linear_historical.csv'
		form = Format(file, training=True)
		self.input_tensors, self.output_tensors = form.sequential_tensors(training=True)
		self.validation_inputs, self.validation_outputs = form.sequential_tensors(training=False)
		self.epochs = epochs

		output_size = 1
		input_size = len(self.input_tensors[0])
		self.model = MultiLayerPerceptron(input_size, output_size)
		# 初始化优化器
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		# 初始化用于绘图的偏置数组
		self.biases_arr = [[], [], []] # for plotting bias
		self.embedding_dim = n_letters # 存储用于模型嵌入层的维度大小，即字符集的长度

	@staticmethod
	def count_parameters(model):
		"""
		Display the tunable parameters in the model of interest

		Args:
			model: torch.nn object

		Returns:
			total_params: the number of model parameters

		"""

		table = PrettyTable(['Modules', 'Parameters'])
		total_params = 0
		for name, parameter in model.named_parameters():
			if not parameter.requires_grad:
				continue
			param = parameter.numel()
			table.add_row([name, param])
			total_params += param 

		print (table)
		print (f'Total trainable parameters: {total_params}')
		return total_params

	def mean_absolute_error(output, target):
		"""
        Calculate the Mean Absolute Error (MAE) between `output` and `target` tensors.

        Args:
            output: torch.Tensor
            target: torch.Tensor

        Returns:
            mae: torch.Tensor
        """
		mae = torch.mean(torch.abs(output - target))
		return mae

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
		if output < target:
			loss = torch.mean((2*(output - target))**2)
		else:
			loss = torch.mean((output - target)**2)

		return loss


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
		# self.model.train()
		output, _ = self.model(input_tensor)
		output_tensor = output_tensor.reshape(minibatch_size, 1)
		loss_function = torch.nn.L1Loss()
		loss = loss_function(output, output_tensor)

		# 梯度清零
		self.optimizer.zero_grad() # prevents gradients from adding between minibatches
		# 反向传播
		loss.backward()

		# 梯度裁剪
		nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
		# 更新模型的参数
		self.optimizer.step()

		return output, loss.item()


	def plot_predictions(self, epoch_number):
		"""
		Plots the model predictions (y-axis) versus the true output (x-axis)

		"""
		self.model.eval() # switch to evaluation mode (silence dropouts etc.) 这一行将模型切换到评估（evaluation）模式。在PyTorch中，模型有两种模式：训练模式和评估模式。在训练模式下，模型会正常使用dropout、批归一化（batch normalization）等只在训练时使用的技术来防止过拟合。而在评估模式下，eval()函数的调用会通知所有的这些层，现在是评估或测试阶段，不应该应用dropout等技术，批归一化层也会使用在训练期间学习到的运行统计数据而不是当前批次的统计数据。
		model_outputs = [] # 这一行初始化一个空列表model_outputs，用来存储模型的输出

		with torch.no_grad():# 这个上下文管理器用于临时禁用梯度计算，减少内存消耗并加速计算。在模型评估或推理时，不需要计算梯度，因为不进行反向传播或模型更新。
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i]
				output_tensor = self.validation_outputs[i]
				model_output, _ = self.model(input_tensor)
				model_outputs.append(float(model_output))

		plt.scatter([float(i) for i in self.validation_outputs], model_outputs, s=1.5)
		_, _, r2, _, _ = scipy.stats.linregress([float(i) for i in self.validation_outputs], model_outputs)
		print (f'R2 value: {r2}')
		plt.axis([-10, 150, -10, 120]) # x-axis range followed by y-axis range
		# plt.show()
		plt.tight_layout()
		plt.savefig('regression_fcnet/regression_fcnet{0:04d}.png'.format(epoch_number), dpi=400)
		plt.close()

		return


	# 将模型中特定层的前几个偏置项（biases）可视化，并通过散点图展示它们的变化趋势或分布
	# 这个散点图展示了特定模型层的前几个偏置值的分布情况或变化趋势。通过观察这些偏置值随时间的变化，可以获取一些关于模型训练过程和模型行为的洞察。例如，如果某组偏置值随时间显著变化，这可能指示着模型在学习过程中对这些神经元的依赖发生了变化。不同颜色的散点代表不同的偏置值组合，可以帮助分析是否有特定的偏置值模式或组合在模型行为中扮演重要角色。
	def plot_biases(self, index):
		"""
		Image model biases as a scatterplot

		Args:
			index: int

		Returns:
			None
		"""
		self.model.eval()
		arr = self.model.hidden2hidden.bias[:6].detach().numpy()
		self.biases_arr[0].append([arr[0], arr[1]])
		self.biases_arr[1].append([arr[2], arr[3]])
		self.biases_arr[2].append([arr[4], arr[5]])
		plt.style.use('dark_background')
		plt.plot([i[0] for i in self.biases_arr[0]], [i[1] for i in self.biases_arr[0]], '^', color='white', alpha=0.7, markersize=0.1)
		plt.plot([i[0] for i in self.biases_arr[1]], [i[1] for i in self.biases_arr[1]], '^', color='red', alpha=0.7, markersize=0.1)
		plt.plot([i[0] for i in self.biases_arr[2]], [i[1] for i in self.biases_arr[2]], '^', color='blue', alpha=0.7, markersize=0.1)
		plt.axis('on')
		plt.savefig('Biases_{0:04d}.png'.format(index), dpi=400)
		plt.close()

		return

	def heatmap_weights(self, index):
		"""
		Plot model weights of one layer as a heatmap

		Args:
			index: int

		Returns:
			None

		"""
		self.model.eval()
		arr = self.model.hidden2hidden2.weight.detach()
		arr = torch.reshape(arr, (2000, 1))
		arr = arr[:44*44]
		arr = torch.reshape(arr, (44, 44))
		arr = arr.numpy()
		plt.imshow(arr, interpolation='spline16', aspect='auto', cmap='inferno')
		plt.style.use('dark_background')
		plt.axis('off')
		plt.savefig('heatmap_{0:04d}.png'.format(index), dpi=400, bbox_inches='tight', pad_inches=0)
		plt.close()
		return


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
			plt.text(x, y, '{:1.3f}'.format(label), ha='center', va='center')

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

		Args:
			None

		Returns:
			None (saves .png)
		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		number_of_examples = 200
		actual_arr, embedding_arr = [], []
		for i in range(number_of_examples):
			input_tensor = self.validation_inputs[i]
			output_tensor = self.validation_outputs[i]
			model_output, embedding = self.model(input_tensor)
			actual_arr.append(float(output_tensor))
			embedding_arr.append(embedding)

		actual_distances, embedding_distances = [], []
		for i in range(len(embedding_arr) - 1):
			for j in range(i, len(embedding_arr)):
				actual_distances.append(np.abs(actual_arr[j] - actual_arr[i]))
				embedding_distance = torch.sum(torch.abs(embedding_arr[j] - embedding_arr[i])).detach().numpy()
				embedding_distances.append(embedding_distance)

		plt.scatter(actual_distances, embedding_distances, s=0.3)
		plt.xlabel('Actual Distance')
		plt.ylabel('Embedding Distance')
		plt.rcParams.update({'font.size': 6})
		# plt.show()
		plt.savefig('embedding_fcnet/embedding_fcnet{0:04d}.png'.format(index), dpi=390)
		plt.close()
		return


	def train_model(self, minibatch_size=32):
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

		for epoch in range(epochs):
			pairs = [[i, j] for i, j in zip(self.input_tensors, self.output_tensors)]
			random.shuffle(pairs)
			input_tensors = [i[0] for i in pairs]
			output_tensors = [i[1] for i in pairs]
			total_loss = 0

			for i in range(0, len(input_tensors) - minibatch_size, minibatch_size):
				# print (count)
				input_batch = torch.stack(input_tensors[i:i + minibatch_size])
				output_batch = torch.stack(output_tensors[i:i + minibatch_size])

				# skip the last batch if it is smaller than the other batches
				if len(input_batch) < minibatch_size:
					break

				output, loss = self.train_minibatch(input_batch, output_batch, minibatch_size)
				total_loss += loss

				#if count % 100 == 0:
				###	self.plot_predictions(count//25) # count//100
					# 这里是25的话：每处理完100个小批量数据，就会绘制一次嵌入图（plot_embedding函数，未提供）和四次回归图（plot_predictions），并且会在特定的迭代中调用StaticInterpret（未提供）来进行模型解释。每轮训练结束后，会打印出总的损失值。
					# 每轮（epoch）中会多次打印embedding图片和regression图片，并多次计算R²值，是因为train_model函数中有基于count变量的条件语句。当count是100的倍数时，会绘制嵌入图；当count是25的倍数时，会绘制回归图和计算R²值。这样设计的目的可能是为了在训练过程中频繁监控模型的性能和学习情况，以便及时调整训练策略。

					#  interpret.heatmap(count, method='combined')
				count += 1

			print (f'Epoch {epoch} complete: {total_loss} loss')
			# self.test_model()

		return


	def test_model(self):
		"""
		Test the model using a validation set

		Args:
			None

		Returns:
			None

		"""

		self.model.eval() # switch to evaluation mode (silence dropouts etc.)
		loss = torch.nn.L1Loss()

		with torch.no_grad():
			total_error = 0
			for i in range(len(self.validation_inputs)):
				input_tensor = self.validation_inputs[i]
				output_tensor = self.validation_outputs[i]
				model_output, _ = self.model(input_tensor)
				total_error += loss(model_output, output_tensor).item()

		print (f'Test Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')

		with torch.no_grad():
			total_error = 0
			for i in range(len(self.input_tensors[:2000])):
				input_tensor = self.input_tensors[i]
				output_tensor = self.output_tensors[i]
				model_output, _ = self.model(input_tensor)
				total_error += loss(model_output, output_tensor).item()

		print (f'Training Average Absolute Error: {round(total_error / len(self.validation_inputs), 2)}')
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
network.test_model()
network.plot_embedding()


















