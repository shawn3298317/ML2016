import pandas as pd
import random

def process_train(train_file, model):

	names = ['dates','items'] + [str(x) for x in range(0,24)]
	# d_clear = {'NR':-1}
	train_raw = pd.read_csv(train_file, names=names, skiprows=[0], usecols=[x for x in range(27) if x!=1])

	# Parse dates to datetimes
	train_raw['dates'] = train_raw['dates'].map(lambda x: pd.to_datetime(x, format="%Y/%m/%d", errors='ignore'))
	# Parse RAINFALL attribute
	train_raw.replace('NR',-1,inplace=True, regex=False)


	# spit to df by dates:
		# group = train_raw.groupby(pd.Grouper(key='dates'))
		# a = train_raw['dates'].value_counts()

	# For Model 1
	if model == 1:
		PM_25 = train_raw[train_raw['items'] == 'PM2.5']
		for row in PM_25.iterrows():
			values = row[1].tolist()[2:] # row_name + data
			for tup in [(float(x),float(y)) for x, y in zip(values[:-1], values[1:])]:
				yield tup #(prev_PM2.5, PM2.5)
	
	# For Model 2
	if model == 2:
		PM_25 = train_raw[train_raw['items'] == 'PM2.5']
		for row in PM_25.iterrows():
			values = [float(i) for i in row[1].tolist()[2:]] # row_name + data

			for tup in [tuple(values[i:i+10]) for i in xrange(len(values)-9)]:
				yield tup

	# For Model 3
	if model == 3:
		# pass
		group = train_raw.groupby(pd.Grouper(key='dates'))
		for gn, gp in group:
			# print gn
			for hr in [str(i) for i in range(23)]:
				
				yield tuple([float(x) for x in gp[hr].tolist()] + [float(x) for x in gp[gp['items'] == 'PM2.5'][str(int(hr)+1)].tolist()])


	# For Model 4
	if model == 4:
		# pass
		group = train_raw.groupby(pd.Grouper(key='dates'))
		for gn, gp in group:
			# print gp
			for i in range(15):
				df = gp[[str(n) for n in range(i,i+10)]]
				# print df
				ret = []
				for col in df:
					if col == str(i+9):
						ret += [float(df[col].tolist()[9])]
					else:
						ret += [float(x) for x in df[col].tolist()]
				
					
				# print df.tolist
				# raw_input()
				# print len(ret)
				yield ret


def process_test(test_file, model):

	names = ['ID', 'items'] + [str(x) for x in range(0,9)]
	test_raw = pd.read_csv(test_file, names = names)
	test_raw.replace('NR',-1,inplace=True, regex=False)

	# For Model 1
	if model == 1:
		PM_25 = train_raw[train_raw['items'] == 'PM2.5']
		for row in PM_25.iterrows():
			values = row[1].tolist()[2:] # row_name + data
			for tup in [(float(x),float(y)) for x, y in zip(values[:-1], values[1:])]:
				yield tup #(prev_PM2.5, PM2.5)
	
	# For Model 2
	if model == 2:
		PM_25 = train_raw[train_raw['items'] == 'PM2.5']
		for row in PM_25.iterrows():
			values = [float(i) for i in row[1].tolist()[2:]] # row_name + data

			for tup in [tuple(values[i:i+10]) for i in xrange(len(values)-9)]:
				yield tup

	# For Model 3
	if model == 3:
		# pass
		group = test_raw.groupby(pd.Grouper(key='ID'))
		for gn, gp in group:
			# print gp
			# raw_input()
			for hr in [str(i) for i in range(8)]:
				
				yield tuple([float(x) for x in gp[hr].tolist()]) #+ [float(x) for x in gp[gp['items'] == 'PM2.5'][str(int(hr)+1)].tolist()])


def genData(numPoints, bias, variance):
    # x = np.zeros(shape=(numPoints, 2))
    # y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(numPoints):
        # bias feature
        # x[i][0] = 1
        # x[i][1] = i
        x = i
        # our target variable
        y = (i + bias)# + random.uniform(0, 1) * variance
    	yield (x, y)



# train_XY = process_train('../data/train.csv', model = 4)
# # test_XY = process_test('../data/test_X.csv', model = 3)
# # train_XY = genData(100, 25, 10)

# for tup in train_XY:
# 	print tup
# 	raw_input() # 18*9+1 = 163




"""
Model 1
y(PM2.5) = w*x(last PM2.5) + b

Model 2
y(PM2.5) = w*x(last 10 hrs' PM2.5) + b

Model 3
y(PM2.5) = w*x(last all items) + b
"""
