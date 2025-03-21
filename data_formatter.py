# data_formatter.py

# import standard libraries
import random
import string

# import third-party libraries
import torch
import pandas as pd 
import numpy as np 


class Format:
	"""
	Formats the 'linear_historical.csv' file to replicate experiments
	格式化' linear_history .csv'文件以复制实验
	in arXiv:2211.02941

	Not intended for general use.
	"""

	# 定义了初始化方法__init__，接收参数：文件路径file，预测特征prediction_feature，取值数n_taken默认为4，和一个布尔值ints_only默认为False。
	# n_taken是用来限制从数据集中选取用于模型训练或分析的特征数量。
	# 它允许你控制在创建模型时考虑的特征数量，从而可以在模型复杂度和计算效率之间做出平衡。
	# 如果n_taken的值较小，那么模型将只使用数据集中的前n_taken个特征，这可能有助于避免过拟合或减少计算时间，但也可能忽略掉一些重要信息。、

	# prediction_feature 在这段代码中指的是数据集中作为预测目标的特征（或列）。
	# 换言之，这是模型试图预测的变量。例如，在泰坦尼克号数据集中，prediction_feature 可能是“Survived”，
	# 表示模型的任务是基于乘客的其他信息来预测他们是否幸存。
	# 这个变量在处理数据、准备训练集和验证集时非常关键，因为它定义了哪个特征应被视为输出（目标变量），哪些特征应被视为输入（特征变量）。
	def __init__(self, file, prediction_feature, n_taken=4, ints_only=False):

		# 使用pd.read_csv(file)读取file指定的CSV文件到DataFrame df
		df = pd.read_csv(file)

		# 对df中的每个元素应用一个函数，如果元素的字符串表示（小写）以'n'开头，则替换为空字符串''。
		# 这行代码遍历DataFrame df的所有元素，并应用一个函数，该函数检查每个元素的字符串表示形式的第一个字符是否为'n'（不区分大小写）。
		# 如果是，该函数将该元素替换为空字符串''；否则，它保留元素不变。
		# 这通常用于数据清洗，特别是在处理包含缺失值标记（如'N', 'n', 'None', 'none'等）的表格数据时，
		# 将这些标记统一替换为空字符串或其他预定义的缺失值表示。
		df = df.applymap(lambda x: '' if str(x).lower()[0] == 'n' else x)
		# 限制df的行数为前100000行。
		df = df[:][:100000]
		# 设置实例变量n_taken
		self.n_taken = n_taken
		# 设置实例变量prediction_feature。
		self.prediction_feature = prediction_feature
		# 通过调用_init_places_dict方法（带参数ints_only）初始化places_dict字典，并将其赋值给实例变量。
		self.places_dict = self._init_places_dict(ints_only)
		# 设置embedding_dim为places_dict的长度。
		self.embedding_dim = len(self.places_dict)

		# 80/20 training/test split and formatting；
		# 计算数据集长度，按照80 / 20
		# 的比例分割数据集为训练集和测试集。
		# 提取训练数据的输入和输出，排除预测特征列作为输入，设置预测特征列作为输出。
		# 重置输入输出的索引，确保从0开始，没有间断。
		length = len(df[:])
		split_i = int(length * 0.8)
		training = df[:][:split_i]
		# 从training数据帧中选择除了预测特征(prediction_feature)以外的所有列。这些列作为模型训练的输入特征
		self.training_inputs = training[[i for i in training.columns if i != prediction_feature]]
		# 从training数据帧中选择预测特征列。这列作为模型训练的输出目标。
		self.training_outputs = training[[prediction_feature]]
		# 分别重置self.training_outputs和self.training_inputs的索引，去掉原来的索引列，确保数据帧的索引从0开始且连续，以便于后续的数据处理和模型训练。
		self.training_outputs.reset_index(inplace=True, drop=True)
		self.training_inputs.reset_index(inplace=True, drop=True)

		# 计算验证集大小，提取验证集的输入和输出，类似于训练集处理，重置索引。
		validation_size = length - split_i
		validation = df[:][split_i:split_i + validation_size]

		self.val_inputs = validation[[i for i in validation.columns if i != prediction_feature]]
		self.val_outputs = validation[[prediction_feature]]
		self.val_inputs.reset_index(inplace=True, drop=True)
		self.val_outputs.reset_index(inplace=True, drop=True)

	# 用于初始化一个将字符映射到嵌入维度张量的字典places_dict。
	# 这是一个私有方法，接受一个布尔参数ints_only，默认为False。

	# 这个函数创建了一个字典places_dict，用于将字符映射到它们的索引值，形成一个字符到嵌入维度张量的映射。
	# 这个映射在处理文本数据时非常有用，特别是当你需要将字符转换为可以被机器学习模型处理的数值形式时。
	# ints_only参数允许选择是只映射数字和一些特殊字符（如果ints_only=True），还是映射所有可打印的ASCII字符（如果ints_only=False）。
	# 这样的映射允许模型能够理解和处理字符数据，为文本数据的进一步分析和模型训练提供基础。
	def _init_places_dict(self, ints_only=False):
		"""
		Initialize the dictionary storing character to embedding dim tensor map.
		初始化字典存储字符为嵌入模糊张量映射。
		kwargs:
			ints_only: bool, if True then a numerical input is expected.
		ints_only: bool，如果为True则需要一个数字输入。

		returns:
			places_dict: dictionary
		places_dict:字典
		"""
		# 如果ints_only为True，则创建一个字典places_dict，
		# 其中包括数字0-9、点号.、空格 、减号-、冒号:和下划线_的枚举，将这些字符映射到它们的索引。
		#
		# 在Python中，enumerate是一个内置函数，它用于将一个可迭代对象（如列表、元组、字符串等）组合为一个索引序列，同时列出数据和数据下标。
		# enumerate通常用在for循环中，使得在循环过程中可以同时获取到元素的索引和值。这在需要通过索引操作元素时非常有用。
		# 例如，enumerate('ABC')会产生一个枚举对象，包含(0, 'A'), (1, 'B'), 和 (2, 'C')这样的元素。
		if ints_only:
			places_dict = {s:i for i, s in enumerate('0123456789. -:_')}
		# 如果ints_only为False，则使用string.printable，
		# 这包括所有可打印的ASCII字符，创建一个字典places_dict，将每个字符映射到其在string.printable中的索引。
		else:
			chars = string.printable
			places_dict = {s:i for i, s in enumerate(chars)}
		# 对于上一行代码的解释
		# 这行代码使用了字典推导式来创建一个字典places_dict，其中包括字符到其索引的映射。
		# enumerate(chars)会为chars中的每个字符生成一个包含索引（i）和值（s）的元组。
		# 字典推导式{s:i for i, s in enumerate(chars)}遍历这些元组，并为每个字符s创建一个条目，其值为该字符在chars中的索引i。
		# 这样，每个字符都被映射到了一个唯一的整数，从而可以用于后续的编码或处理过程中。
		return places_dict

	# 下面的函数用于将数据集中的行转换为字符串格式，以便用于模型训练或测试。
	# 函数接受三个参数：input_type（数据类型，如训练、测试或验证数据）、short（是否对每个特征最多使用 n_taken 个字母）、remove_spaces（是否在编码前删除输入特征的空格）。
	def stringify_input(self, input_type='training', short=True, remove_spaces=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistant structure to inputs regardless of missing values, with n_taken characters per field.

		kwargs:
			input_type: str, type of data input requested ('training' or 'test' or 'validation')
			short: bool, if True then at most n_taken letters per feature is encoded
			remove_spaces: bool, if True then input feature spaces are removed before encoding

		Returns:
			array: string: str of values in the row of interest

		"""
		# n 和 n_taken 被设置为类实例的 n_taken 属性，决定了每个字段取值的字符数。
		n = n_taken = self.n_taken
		# 根据 input_type 的值，选择相应的数据输入（训练training、验证validation或测试test）
		if input_type == 'training':
			inputs = self.training_inputs

		elif input_type == 'validation':
			inputs = self.val_inputs

		else:
			inputs = self.test_inputs

		# short指的是是否对每个特征最多使用 n_taken 个字母。
		# 如果 short 为 True，则对输入应用一个函数，将空字符串或 '(null)' 替换为 n_taken 个下划线，其它值则取前 n 个字符。
		# 如果 short 为 False，则保留完整字符串，但仍替换空字符串或 '(null)'。
		if short == True:
			inputs = inputs.applymap(lambda x: '_'*n_taken if str(x) in ['', '(null)'] else str(x)[:n])
		else:
			inputs = inputs.applymap(lambda x:'_'*n_taken if str(x) in ['', '(null)'] else str(x))
		# 通过另一个 applymap 调用，确保每个值的长度达到 n_taken，不足部分用下划线填充。
		# 使用 apply 函数和 join 方法，将每一行的值连接成一个由下划线分隔的字符串。
		#
		# 对于输入数据的每个元素，该行代码首先计算每个值的长度与n_taken的差值，
		# 然后在每个值的前面填充相应数量的下划线_，以确保每个值的总长度等于n_taken。
		# 这样做的目的是为了保持数据的一致性，确保模型可以处理长度一致的输入。
		inputs = inputs.applymap(lambda x: '_'*(n_taken - len(x)) + x)
		#
		# 这一行将每行的数据值转换为字符串（如果它们还不是字符串），然后用下划线_连接这些字符串值，生成一个新的字符串。
		# 这样，每行数据都被转换成了一个由下划线分隔的单一字符串，方便后续的模型处理。
		# axis=1指明这个操作是沿着行进行的，即对每一行内的元素执行这个操作。
		string_arr = inputs.apply(lambda x: '_'.join(x.astype(str)), axis=1)
		return string_arr

	# 将字符串转换为张量
	def string_to_tensor(self, string, flatten):
		"""
		Convert a string into a tensor

		Args:
			string: arr[str]
			flatten: bool, if True then tensor has dim [1 x length]

		Returns:
			tensor: torch.Tensor

		"""
		# 此函数接收一个字符串数组和一个布尔值 flatten。
		# 它首先创建一个形状为字符串长度乘以词汇表大小的零张量。
		# 然后，对于字符串中的每个字符，它在张量的相应位置上设置为 1，根据字符在词汇表中的索引。
		# 如果 flatten 为真，则将张量展平。
		places_dict = self.places_dict

		# vocab_size x embedding dimension (ie input length)
		tensor_shape = (len(string), len(places_dict)) 
		tensor = torch.zeros(tensor_shape)

		for i, letter in enumerate(string):
			tensor[i][places_dict[letter]] = 1.

		if flatten:
			tensor = torch.flatten(tensor)
		return tensor

	# 此函数根据是训练数据还是验证数据，调用 stringify_input 函数来获取字符串数组，并获取相应的输出值。
	# 然后，它遍历字符串数组，将每个字符串及其对应的输出值转换为张量，并将这些张量分别添加到输入和输出数组中。
	def transform_to_tensors(self, training=True, flatten=True): # training 用于指定是否为训练数据，flatten 指定是否将张量展平。
		"""
		Transform input and outputs to arrays of tensors

		kwargs:
			flatten: bool, if True then tensors are of dim [1 x length]

		"""
		# 根据 training 参数的值，选择是使用训练数据 (self.training_outputs) 还是验证数据 (self.val_outputs)。
		# 通过调用 self.stringify_input 方法，根据输入类型（训练或验证），获取转换后的字符串数组。
		# 同时，根据是训练还是验证状态，从相应属性中获取输出数据。
		if training:
			string_arr = self.stringify_input(input_type='training')
			outputs = self.training_outputs

		else:
			string_arr = self.stringify_input(input_type='validation')
			outputs = self.val_outputs

		# 初始化两个空数组 input_arr 和 output_arr，用于存储最终的输入和输出张量。
		input_arr, output_arr = [], []
		# 通过 for 循环遍历 string_arr，对于每个元素：
		# 条件检查：检查对应的输出值是否非空。如果非空，说明这个数据点是有效的，可以进行转换。
		# 字符串转张量：调用 self.string_to_tensor 方法，将字符串转换为张量。如果 flatten 为 True，则将得到的张量展平。
		# 收集输入和输出张量：将转换后的输入张量添加到 input_arr 中，将输出值转换为张量后添加到 output_arr 中。
		for i in range(len(string_arr)):
			
			if outputs[self.prediction_feature][i] != '':
				string = string_arr[i]
				input_arr.append(self.string_to_tensor(string, flatten))
				output_arr.append(torch.tensor(outputs[self.prediction_feature][i]))

		# 函数返回两个数组，input_arr 包含了所有输入的张量，output_arr 包含了所有输出的张量。
		return input_arr, output_arr

	# 此函数遍历测试输入数据集，使用 string_to_tensor 函数将每个测试输入字符串转换为张量，然后将这些张量添加到一个数组中，最后返回这个数组。
	# 这个函数好像没有被调用
	def generate_test_inputs(self):
		"""
		Generate tensor inputs from a test dataset

		"""
		inputs = []
		for i in range(len(self.test_inputs)):
			# 原本是input_tensor =self.string_to_tensor(input_string)
			input_tensor =self.string_to_tensor(input_string)
			# 修改版：input_tensor =self.string_to_tensor(self.input_string)
			inputs.append(input_tensor)

		return inputs


# FormatDeliveries 类是专门为处理和转换快递相关数据而设计的。
# 实际上它是Format类的一个特化，专门优化了对某类数据的处理。
# 它继承了Format类的基本功能，同时添加了针对特定数据字段的处理方法，如stringify_input和unstructured_stringify，这两个方法都是针对特定数据集字段的处理而设计。
# 特别地，FormatDeliveries类通过提供更细致的字段处理方式（例如，针对每个字段可以设置不同的字符取值数），使得数据预处理更加灵活和精细。
class FormatDeliveries:
	"""data_formatter
	Formats the 'linear_historical.csv' file to replicate experiments
	in arXiv:2211.02941

	Not intended for general use.
	"""

	def __init__(self, file, prediction_feature, training=True, n_per_field=False):
		# 根据n_per_field参数的值，设置每个字段取值的字符数。
		# n_per_field为True时，为所有输入字段分配统一的字符数（这里是4个字符）；
		# 为False时，则为每个字段指定不同的字符数，这体现在self.taken_ls的设定上。

		df = pd.read_csv(file)	
		df = df.applymap(lambda x: '' if str(x).lower() == 'nan' else x)
		df = df[:10000]
		length = len(df['Elapsed Time'])
		self.input_fields = ['Store Number', 
							'Market', 
							'Order Made',
							'Cost',
							'Total Deliverers', 
							'Busy Deliverers', 
							'Total Orders',
							'Estimated Transit Time',
							'Linear Estimation']

		if n_per_field:
			self.taken_ls = [4 for i in self.input_fields] # somewhat arbitrary size per field
		else:
			self.taken_ls = [4, 1, 8, 5, 3, 3, 3, 4, 4]

		if training:
			# df = shuffle(df)
			df.reset_index(inplace=True)
 
			# 80/20 training/validation split
			split_i = int(length * 0.8)

			training = df[:][:split_i]
			self.training_inputs = training[self.input_fields]
			self.training_outputs = [i for i in training[prediction_feature][:]]

			validation_size = length - split_i
			validation = df[:][split_i:split_i + validation_size]
			self.validation_inputs = validation[self.input_fields]
			self.validation_outputs = [i for i in validation[prediction_feature][:]]
			self.validation_inputs = self.validation_inputs.reset_index()

		else:
			self.training_inputs = self.df # not actually training, but matches name for stringify

	# 将特定行的数据转换成一个字符串数组，每个字段的字符数由 taken_ls 列表指定。
	def stringify_input(self, index, training=True):
		"""
		Compose array of string versions of relevant information in self.df 
		Maintains a consistent structure to inputs regardless of missing values.

		Args:
			index: int, position of input n 

		Returns:
			array: string: str of values in the row of interest

		"""
		# 初始化过程
		taken_ls = self.taken_ls
		string_arr = []
		# 根据 training 参数选择训练或验证输入数据
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]

		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])[:taken_ls[i]]
			while len(entry) < taken_ls[i]:
				entry += '_'
			string_arr.append(entry)
		# 使用 join 方法将 string_arr 中的所有元素连接成一个单一字符串
		string = ''.join(string_arr)
		return string

	# 用于将指定行的数据转换成一个未结构化的字符串，可选地进行填充以达到指定长度。
	# unstructured_stringify 接收四个参数：
	# index（整型，指定行的索引），training（布尔值，指示使用训练数据还是验证数据），
	# pad（布尔值，指示是否填充字符串到指定长度），和 length（整型，指定填充后字符串的长度，默认为50）。
	def unstructured_stringify(self, index, training=True, pad=True, length=50):
		"""
		Compose array of string versions of relevant information in self.df 
		Does not maintain a consistant structure to inputs regardless of missing 
		values.

		Args:
			index: int, position of input

		Returns:
			array: string: str of values in the row of interest

		"""

		# string_arr 初始化为空列表，用于存储转换后的字符串元素。
		string_arr = []
		if training:
			inputs = self.training_inputs.iloc[index]
		else:
			inputs = self.validation_inputs.iloc[index]

		# 通过遍历 self.input_fields（字段列表），将每个字段的值转换为字符串并添加到 string_arr 中。
		fields_ls = self.input_fields
		for i, field in enumerate(fields_ls):
			entry = str(inputs[field])
			string_arr.append(entry)

		# 使用 join 方法将 string_arr 中的所有字符串元素拼接成一个单一的字符串 string。
		string = ''.join(string_arr)
		# 如果 pad 为 True，则根据 string 的长度与指定的 length 进行比较，以决定是否需要填充或截断：
		#
		# 如果 string 的长度小于 length，则在 string 的末尾添加足够数量的下划线(_)，使其长度达到 length。
		# 如果 string 的长度大于 length，则将 string 截断为前 length 个字符。
		if pad:
			if len(string) < length:
				string += '_' * (length - len(string))
			if len(string) > length:
				string = string[:length]

		return string


	@classmethod
	# 将一个字符串转换为一个张量，主要用于文本数据的数值化处理，以便用于深度学习模型
	def string_to_tensor(self, input_string):
		"""
		Convert a string into a tensor

		Args:
			string: str, input as a string

		Returns:
			tensor: torch.Tensor() object
		"""

		# 定义了一个字典 places_dict，该字典将数字字符 '0' 到 '9' 映射到它们的整数值。
		# 这是为了能够将字符串中的数字字符转换为对应的整数。
		places_dict = {s:int(s) for s in '0123456789'}
		# 通过遍历一个包含特殊字符的字符串 '. -:_'，将这些特殊字符也加入到 places_dict` 字典中，并给它们分配一个唯一的整数值。
		# 这些特殊字符被分配的整数从10开始，以确保不与数字字符的映射冲突。
		for i, char in enumerate('. -:_'):
			places_dict[char] = i + 10
		# 确定张量的形状 tensor_shape 为 (len(input_string), 1, 15)。
		# 这里，len(input_string) 表示输入字符串的长度，1是批次大小（这里处理的是单个字符串，所以批次大小为1），
		# 15是嵌入维度，代表了数字和特殊字符总共可能的不同值（10个数字加上5个特殊字符）。

		# vocab_size x batch_size x embedding dimension (ie input length)
		tensor_shape = (len(input_string), 1, 15)
		# 创建一个形状为 tensor_shape 的全零张量 tensor。这个张量用于存储字符串每个字符的one-hot编码。
		tensor = torch.zeros(tensor_shape)
		# 遍历输入字符串的每个字符，根据 places_dict 字典找到每个字符对应的整数索引，然后在张量的相应位置设置为1。
		# 这实现了将每个字符转换为one-hot编码的过程。
		for i, letter in enumerate(input_string):
			tensor[i][0][places_dict[letter]] = 1.
		# 使用 tensor.flatten() 将张量展平，这是为了将多维张量转换为一维，以便于后续处理。
		tensor = tensor.flatten()
		return tensor 

	# 将输入和输出数据转换成张量形式，用于训练或验证模型。
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

 
