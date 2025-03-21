# linear_models.py

import pandas as pd 
import numpy as np 

import sklearn 
from sklearn.utils import shuffle
from datetime import datetime
from statsmodels.formula.api import ols

# turn 'value set on df slice copy' warnings off
pd.options.mode.chained_assignment = None

# 将CSV文件('historical_data.csv')中的日期数据重新格式化为日期时间对象，生成formatted_historical.csv
# 计算'actual_delivery_time'和'created_at'之间以秒为单位的经过时间，并返回带有经过时间新列的DataFrame
def format_data():
	"""
	Reformats date data into datetime objects

	Args:
		None

	Returns:
		None (saves 'formatted_historical.csv')

	"""
	file = 'historical_data.csv'
	df = pd.read_csv(file)
	df = shuffle(df)

	# remove rows with null times
	df.reset_index(inplace=True)

	arr = []
	# convert datetime strings to datetime objects
	for i in range(len(df['actual_delivery_time'])):

		# Bottleneck here: strptime string matching is rather slow. 
		# Expect for prod data datetimes to already be formatted if pulled from an SQL database
		deldate = datetime.strptime(str(df['actual_delivery_time'][i]), '%Y-%m-%d %H:%M:%S')
		credate = datetime.strptime(str(df['created_at'][i]), '%Y-%m-%d %H:%M:%S')

		df['actual_delivery_time'][i] = deldate
		df['created_at'][i] = credate
		elapsed_time = df['actual_delivery_time'][i] - df['created_at'][i]
		elapsed_seconds = elapsed_time.total_seconds()

		arr.append(elapsed_seconds)

	# create column of actual wait time from start and end datetimes
	df['etime'] = arr

	return df

# format_data()
df = pd.read_csv('data/formatted_historical.csv')

# 使用'store_id'， 'market_id'等特征执行多元线性回归来预测持续时间('etime')。
# 它将数据分成训练集和验证集，使用训练集拟合模型，用验证集评估模型，并打印出各种误差测量值
def linear_regression(df):
	"""
	Perform a multiple linear regression to predict duration

	Args:
		df: pd.dataframe

	Returns:
		fit: statsmodel.formula.api.ols.fit() object
		None (prints error measurements)

	"""
	df = df.dropna(axis=0)

	# 计算DataFrame的长度。
	# 根据数据总数的80%计算训练集的索引界限。
	# 根据计算出的索引划分数据为80%的训练集和20%的验证集。
	length = len(df['etime']) # 这行代码移除DataFrame中包含空值的行，以确保后续分析中不会因缺失数据引发错误。
	split_i = int(length * 0.8) # 80/20 training/test split
	training = df[:][:split_i]
	validation_size = length - split_i
	validation = df[:][split_i:split_i + validation_size]


	# 提取验证集中的特定列作为输入变量，以及 etime 列作为输出变量（即预测目标）
	val_inputs = validation[['store_id', 
							'market_id', 
							'total_busy_dashers', 
							'total_onshift_dashers', 
							'total_outstanding_orders',
							'estimated_store_to_consumer_driving_duration']]
	val_outputs = [i for i in validation['etime']]

	# 使用statsmodels库的 ols 函数根据指定的公式拟合线性回归模型。这里使用 C(market_id) 表示 market_id 是分类变量。
	# R-like syntax
	fit = ols('etime ~ \
			   C(market_id) + \
			   total_onshift_dashers + \
			   total_busy_dashers + \
			   total_outstanding_orders + \
			   estimated_store_to_consumer_driving_duration', data=training).fit() 


	print (fit.summary())
	predictions = fit.predict(val_inputs)

	# weighted MSE and MAE accuracy calculations
	count = 0
	loss = 0
	ab_loss = 0
	weighted_loss = 0
	for i, pred in enumerate(predictions):	
		target = val_outputs[i]
		output = pred
		ab_loss += abs(output - target)
		if output < target:
			weighted_loss += 2*abs(output - target)
		else:
			weighted_loss += abs(output - target)
		if output < target:
			loss += (2*(output - target))**2
		else:
			loss += (output - target)**2
		count += 1

	weighted_mse = loss / count
	print (f'weighted absolute error: {weighted_loss / count}')
	print (f'Mean Abolute Error: {ab_loss / count}')
	print (weighted_mse)
	print ('weighted RMS: {}'.format(weighted_mse**0.5))

	mse = sum([(i-j)**2 for i, j in zip(predictions, val_outputs)]) / validation_size
	print (mse)
	print ('Linear model RMS error: ', (mse**0.5))

	return fit


fit = linear_regression(df)

# 使用训练好的线性模型来预测新数据集('predict_data.csv')上的值，将预测添加到DataFrame，并返回它
def linear_fit(fit, file_name):
	"""
	Predict values with the trained linear model

	Args:
		fit: statsmodel.formula.api.ols.fit() object

	Returns:
		df: pd.dataframe

	"""

	df = pd.read_csv('data/predict_data.csv')
	length = len(df['market_id'])

	estimations = fit.predict(df)
	print (estimations)

	df['linear_ests'] = estimations
	df.to_csv('data/modified_predict_data.csv', index=False)
	return df


file_name = 'data/predict_data.csv'
df = linear_fit(fit, file_name)

# 将DataFrame中指定的float字段转换为整数，在适当的位置修改DataFrame，并将其保存到指定的文件名
def convert_to_int(df, field, file_name):
	"""
	Convert a float field to ints

	Args:
		df: pandas dataframe
		field: str

	Returns:
		none (modifies df and saves to storage)

	"""


	for i in range(len(df[field])):
		if str(df[field][i]) not in ['', 'NaN', 'nan']:
			df[field][i] = int(df[field][i])

	df.to_csv(file_name)


file_name = 'data_to_predict.csv'
field = 'linear_ests'
convert_to_int(df, field, file_name)


