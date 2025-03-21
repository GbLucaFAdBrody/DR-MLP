# transformer.py

# import standard libraries
import string
import time
import math
import random

# import third-party libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.utils import shuffle
from statsmodels.formula.api import ols
from prettytable import PrettyTable

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib
import torch.optim as optim
from network_interpret import StaticInterpret

# import custom libraries
from network_interpret import Interpret 
from data_formatter_2 import Format
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 这是PyTorch中的一个函数，用于指定张量计算应该在哪个设备上执行。它的参数可以是字符串如'cuda'或'cpu'
# Turn 'value set on df slice copy' warnings off, but
# note that care should be taken to match pandas dataframe
# column to the appropriate type
pd.options.mode.chained_assignment = None


class Transformer(nn.Module):
	"""
	Encoder-only tranformer architecture for regression.  The approach is 
	to average across the states yielded by the transformer encoder before
	passing this to a single hidden fully connected linear layer.
	"""
	def __init__(self, output_size, line_length, n_letters, nhead, feedforward_size, nlayers, minibatch_size, dropout=0.3, posencoding=False):

		super().__init__()# 用于调用父类（或超类）的构造方法(这里是nn.Module)
		self.posencoder = PositionalEncoding(n_letters)
		encoder_layers = TransformerEncoderLayer(n_letters, nhead, feedforward_size, dropout, batch_first=True)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.transformer2hidden = nn.Linear(line_length * n_letters, 50)
		self.hidden2output = nn.Linear(50, 1)
		self.relu = nn.ReLU()
		self.posencoding = posencoding




	def forward(self, input_tensor):
		"""
		Forward pass through network

		Args:
			input_tensor: torch.Tensor of character inputs

		Returns: 
			output: torch.Tensor, linear output
		"""

		# apply (relative) positional encoding if desired
		if self.posencoding:
			input_encoded = self.posencoder(input_tensor)

		output = self.transformer_encoder(input_tensor)

		# output shape: same as input (batch size x sequence size x embedding dimension)
		output = torch.flatten(output, start_dim=1)
		output = self.transformer2hidden(output)
		output = self.relu(output)
		output = self.hidden2output(output)

		# return linear-activation output
		return output

# 使用的位置编码方法是基于正弦和余弦函数的位置编码
# 这种位置编码方式非常有效，因为它允许模型即使在处理非常长的序列时，也能够区分出不同位置的输入元素。正弦和余弦函数的使用保证了编码的连续性和周期性，这对于很多基于序列的任务（如文本处理、时间序列分析等）是非常有益的。此外，这种方法不依赖于序列的绝对位置，使得模型能够更好地泛化到不同长度的输入上。
class PositionalEncoding(nn.Module):
	"""
	Encodes relative positional information on the input
	"""

	def __init__(self, model_size, max_len=1000):

		super().__init__()
		self.model_size = model_size
		if self.model_size % 2 == 0:
			arr = torch.zeros(max_len, model_size)

		else:
			arr = torch.zeros(max_len, model_size + 1)

		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, model_size, 2).float() * (-math.log(10*max_len) / model_size))
		arr[:, 0::2] = torch.sin(position * div_term)
		arr[:, 1::2] = torch.cos(position * div_term)
		arr = arr.unsqueeze(0)
		self.arr = arr


	def forward(self, tensor):
		"""
		Apply positional information to input

		Args:
			tensor: torch.Tensor, network input

		Returns:
			dout: torch.Tensor of modified input

		"""
		tensor = tensor + self.arr[:, :tensor.size(1), :tensor.size(2)]
		return tensor


class ActivateNetwork:

	def __init__(self):
		embedding_dim = len('0123456789. -:_')
		file = 'data/linear_historical.csv'
		input_tensors = Format(file, 'positive_three')

		self.train_inputs, self.train_outputs = input_tensors.transform_to_tensors(training=True, flatten=False)
		self.test_inputs, self.test_outputs = input_tensors.transform_to_tensors(training=False, flatten=False)
		self.line_length = len(self.train_inputs[0])
		print (self.line_length)
		self.minibatch_size = 32
		self.model = self.init_transformer(embedding_dim, self.minibatch_size, self.line_length)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
		self.loss_function = nn.L1Loss()
		self.epochs = 200

	# 函数参数：
	# embedding_dim: 输入的嵌入维度，也被用作模型中的 d_model，即每个输入token的特征向量的维度。
	# minibatch_size: 模型训练时每个小批次的大小。
	# line_length: 输入序列的长度，即模型需要处理的每个输入的token数。
	#
	# Transformer模型的参数
	# n_output: 输出的维度，这里设置为1，通常用于回归或二分类任务。
	# n_letters: 设置为 embedding_dim，在这里表示模型的特征维度。
	# nhead: 多头注意力机制的头数，这里设为5。注意头数需要能够整除 d_model，以确保每个头可以均等地分配到特征。
	# feedforward_size: 前馈全连接层的维度，这里设置为280。
	# nlayers: Transformer编码器中的层数，这里使用了3层。
	#
	# 三层，每一层都包括如下三个组成部分：
	# 多头自注意力机制：
	# 这一机制允许模型在处理每个序列元素时同时关注序列中的其他元素。这是通过计算不同的查询（Query）、键（Key）和值（Value）表示来实现的。多头自注意力通过并行地学习数据的不同子空间表示，增强了模型对信息的整合能力。
	# 前馈全连接网络：
	# 每个编码器层中的自注意力模块的输出会传递到一个前馈全连接网络。这个网络通常包含两个线性变换层，它们之间有一个非线性激活函数，如ReLU。这个网络在每个位置上独立地对数据进行处理，即它不会改变数据的序列结构，只是对每个元素进行变换。
	# 残差连接和层归一化：
	# 每个子层（自注意力和前馈网络）的输出都添加一个从该子层输入直接来的残差连接，然后进行层归一化。残差连接帮助模型在很深的网络结构中维持有效的梯度流，层归一化则用于调整层的输出，使其均值和方差标准化，这有助于稳定训练过程。
	#
	# 为什么弄成三层：
	# 1. 模型容量和复杂性
	# 更多的层数意味着更高的模型复杂性。每增加一层，模型就能学习更复杂的特征表示，从而在理论上能更好地理解和处理复杂的输入数据。然而，层数的增加也会带来更多的参数，这可能导致过拟合，尤其是在数据量较少的情况下。因此，选择三层可能是一种在模型复杂性和防止过拟合之间的平衡。
	# 2. 计算效率和训练成本
	# 每增加一层，所需的计算资源和训练时间也会增加。在有限的资源下，增加太多层可能导致训练过程变得不切实际。选择三层可能是基于可用资源和预期的训练效率做出的折衷决定。
	# 3. 任务相关性
	# 不同的任务可能需要不同层级的抽象和特征学习能力。对于一些较为简单的任务，使用较少的层数就已足够，而对于需要深层次语义理解的复杂任务（如机器翻译或文本生成），可能需要更多的层数。选择三层可能是根据特定任务的复杂性和需求来确定的。
	# 4. 经验和实验结果
	# 在实践中，模型的最佳层数往往是通过实验确定的。研究者们可能会尝试不同的层数，通过比较模型在验证集上的表现来选择最佳的配置。选择三层可能基于先前的实验结果表明，这一配置在当前任务和数据集上提供了最好的性能和泛化能力。
	# 5. 理论和前人研究
	# 有些时候，选择特定层数也可能受到理论研究或前人经验的影响。例如，早期的Transformer模型研究可能已经表明，在特定类型的任务上，使用三层可以达到一个较好的性能平衡点。


	# 初始化transformer
	def init_transformer(self, embedding_dim, minibatch_size, line_length):
		"""
		Initialize a transformer model

		Args:
			n_letters: int, number of ascii inputs
			emsize: int, the embedding dimension

		Returns:
			model: Transformer object

		"""

		# note that nhead (number of multi-head attention units) must be able to divide d_model
		feedforward_size = 280
		nlayers = 3
		nhead = 5
		n_letters = embedding_dim # set the d_model to the number of letters used # 也是d_model
		n_output = 1
		model = Transformer(n_output, line_length, n_letters, nhead, feedforward_size, nlayers, minibatch_size)
		return model

	# 回归/预测
	def plot_predictions(self, validation_inputs, validation_outputs, count):
		"""
		Plots the model's predicted values (y-axis) against the true values (x-axis)

		Args:
			model: torch.nn.Transformer module
			validation_inputs: arr[torch.Tensor] 
			validations_outputs: arr[torch.Tensor]
			count: int, iteration of plot in sequence

		Returns:
			None (saves png file to disk)
		"""

		self.model.eval()
		model_outputs = []

		with torch.no_grad():
			total_error = 0
			for i in range(len(validation_inputs)):
				input_tensor = validation_inputs[i]
				input_tensor = input_tensor.reshape(1, len(input_tensor), len(input_tensor[0]))
				output_tensor = validation_outputs[i]
				model_output = self.model(input_tensor)
				model_outputs.append(float(model_output))

		plt.scatter([float(i) for i in validation_outputs], model_outputs, s=1.5)
		plt.axis([-10, 150, -10, 120])  # x-axis range followed by y-axis range
		# [-10, 150, -10, 120]
		# plt.show()
		plt.xlabel('Actual Output', fontsize=12)
		plt.ylabel('Expected Output', fontsize=12)
		plt.tick_params(axis='both', which='major', labelsize=12)
		plt.tight_layout()
		plt.savefig('regression_transformer/regression_transformer{0:04d}.png'.format(count), dpi=400)
		plt.close()
		return

	# 计算参数数量
	def count_parameters(self):
		"""
		Display the tunable parameters in the model of interest

		Args:
			model: torch.nn object

		Returns:
			total_params: the number of model parameters

		"""

		table = PrettyTable(['Modules', 'Parameters'])
		total_params = 0
		for name, parameter in self.model.named_parameters():
			if not parameter.requires_grad:
				continue
			param = parameter.numel()
			table.add_row([name, param])
			total_params += param 

		print (table)
		print (f'Total trainable parameters: {total_params}')
		return total_params


	def quiver_gradients(self, index, input_tensor, output_tensor, minibatch_size=32):
			"""
			Plot a quiver map of the gradients of a chosen layer's parameters

			Args:
				index: int, current training iteration
				model: pytorch transformer model
				input_tensor: torch.Tensor object
				output_tensor: torch.Tensor object
			kwargs:
				minibatch_size: int, size of minibatch

			Returns:
				None (saves matplotlib pyplot figure)
			"""
			model = self.model
			model.eval()
			layer = model.transformer_encoder.layers[0]
			x, y = layer.linear1.bias[:2].detach().numpy()
			print (x, y)
			plt.style.use('dark_background')

			x_arr = np.arange(x - 0.01, x + 0.01, 0.001)
			y_arr = np.arange(y - 0.01, y + 0.01, 0.001)

			XX, YY = np.meshgrid(x_arr, y_arr)
			dx, dy = np.meshgrid(x_arr, y_arr) # copy that will be overwritten
			for i in range(len(x_arr)):
				for j in range(len(y_arr)):
					with torch.no_grad():
						layer.linear1.bias[0] = torch.nn.Parameter(torch.Tensor([x_arr[i]]))
						layer.linear1.bias[1] = torch.nn.Parameter(torch.Tensor([y_arr[j]]))
					model.transformer_encoder.layers[0] = layer
					output = model(input_tensor)
					output_tensor = output_tensor.reshape(minibatch_size, 1)
					loss_function = torch.nn.L1Loss()
					loss = loss_function(output, output_tensor)
					optimizer = optim.Adam(model.parameters(), lr=0.001)
					optimizer.zero_grad()
					loss.backward()
					layer = model.transformer_encoder.layers[0]
					dx[j][i], dy[j][i] = layer.linear1.bias.grad[:2]

			matplotlib.rcParams.update({'font.size': 8})
			color_array = 2*(np.abs(dx) + np.abs(dy))
			plt.quiver(XX, YY, dx, dy, color_array)
			plt.plot(x, y, 'o', markersize=1)
			plt.savefig('quiver_{0:04d}.png'.format(index), dpi=400)
			plt.close()
			with torch.no_grad():
				model.transformer_encoder.layers[0].linear1.bias.grad[:2] = torch.Tensor([x, y])
			return

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

		self.model.train()
		output = self.model(input_tensor)
		output_tensor = output_tensor.reshape(minibatch_size, 1)

		loss = self.loss_function(output, output_tensor)
		self.optimizer.zero_grad() # prevents gradients from adding between minibatches
		loss.backward()

		nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
		self.optimizer.step()

		return output, loss.item()


	def train_model(self):
		"""
		Train the transformer encoder-based model

		Args:
			model: MultiLayerPerceptron object
			optimizer: torch.optim object

		kwargs:
			minibatch_size: int

		Returns:
			None

		"""

		self.model.train()
		count = 0
		losses = []
		for epoch in range(self.epochs):
			pairs = [[i, j] for i, j in zip(self.train_inputs, self.train_outputs)]
			random.shuffle(pairs)
			input_tensors = [i[0] for i in pairs]
			output_tensors = [i[1] for i in pairs]
			total_loss = 0

			for i in range(0, len(pairs) - self.minibatch_size, self.minibatch_size):
				# stack tensors to make shape (minibatch_size, input_size)
				input_batch = torch.stack(input_tensors[i:i + self.minibatch_size])
				output_batch = torch.stack(output_tensors[i:i + self.minibatch_size])

				# skip the last batch if too small
				if len(input_batch) < self.minibatch_size:
					break

				# tensor shape: batch_size x sequence_len x embedding_size
				output, loss = self.train_minibatch(input_batch, output_batch, self.minibatch_size)
				total_loss += loss
				losses.append(loss)
				count += 1
				if count % 100 == 0: # plot every 23 epochs for minibatch size of 32
					self.plot_predictions(self.test_inputs, self.test_outputs, count//100)
					# self.quiver_gradients(self.model, input_batch, output_batch)

			print (f'Epoch {epoch} complete: {total_loss} loss')
			# 绘制损失值图形
		plt.figure(figsize=(10, 5))
		plt.plot(losses, label='Loss per Batch')
		plt.xlabel('Batch Number')
		plt.ylabel('Loss')
		plt.title('Loss per Batch During Training')
		plt.savefig('loss_transformer/loss_transformer.png', dpi=400)
		plt.legend()
		plt.show()
		return

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


	def save_model(self, model):
		"""
		Saves a Transformer object state dictionary

		Args:
			model: Transformer class object

		Returns:
			None

		"""

		file_name = 'transformer.pth'
		torch.save(model.state_dict(), file_name)
		return

	@torch.no_grad()
	def evaluate_network(self):
		"""
        Evaluate network on validation data.

        Args:
            None

        Returns:
            None (prints accuracies)

        """
		model = self.model
		model.eval()  # switch to evaluation mode (silence dropouts etc.)
		count = 0
		validation_inputs = self.test_inputs
		validation_outputs = self.test_outputs

		squared_error = 0
		mae_error = 0
		for i in range(len(validation_inputs) // self.minibatch_size):
			input_batch = torch.stack(validation_inputs[i:i + self.minibatch_size]).to(device)
			output_batch = torch.stack(validation_outputs[i:i + self.minibatch_size]).to(device)
			model_output, *_ = model(input_batch)
			squared_error += torch.sum((model_output - output_batch) ** 2).item()
			mae_error += torch.sum(torch.abs(model_output - output_batch)).item()
			count += 1

		rms_error = (squared_error / count) ** 0.5
		print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
		print(f'Validation RMS error: {round(rms_error, 2)} \n')
		print(f'Validation Mean Absolute Error: {mae_error / count}')

		return

network = ActivateNetwork()
network.train_model()
network.evaluate_network()



