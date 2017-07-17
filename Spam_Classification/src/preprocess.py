import numpy as np


def process_train(train_file):
    
    with open(train_file, 'r') as f:
        train_X = []
        train_Y = []
        for line in f:
            tok = line.split(',')
            train_X.append([float(x) for x in tok[1:-1]]) #[int(tok[0])] + 
            train_Y.append(np.eye(2)[int(tok[-1])])
            # print line
        #     print train_X[-1]
        #     print train_Y[-1]
        #     raw_input()
        # print len(train_X)
        # print len(train_Y)
        return train_X[:-1], train_Y[:-1]

def process_test(test_file):
    
    with open(test_file, 'r') as f:
        test_X = []
        for line in f:
            tok = line.split(',')
            test_X.append([float(x) for x in tok[1:]]) #[int(tok[0])] + 
            # print line
        #     print train_X[-1]
        #     print train_Y[-1]
        #     raw_input()
        # print len(train_X)
        # print len(train_Y)
        return test_X[:]

def data_split_4_cross_valid(train_X, train_Y):
    train_x_split = {}
    train_y_split = {}

    for n in range(10): # 10-fold
        # print n
        train_x_split[n] = train_X[400*n:400*(n+1)]
        train_y_split[n] = train_Y[400*n:400*(n+1)]
        # print len(train_x_split[-1])
        # print np.array(train_x_split[n][0])
        # raw_input()
    return train_x_split, train_y_split

# Perform K-fold splitting and return the corresponding train and test batches for current epoch
def get_batch_data(fold_idx):

    x_test_batches = train_x_split[fold_idx]
    # print len(x_test_batches), len(x_test_batches[0]), len(x_test_batches[0][0])
    # raw_input()
    y_test_batches = train_y_split[fold_idx]
    x_train_batches = []
    y_train_batches = []
    to_concat = range(10)
    to_concat.remove(fold_idx)
    for x in to_concat:
        for yx, yy in zip(train_x_split[x], train_y_split[x]):
            x_train_batches.append(yx)
            y_train_batches.append(yy)
    # print "Batch size: ", len(x_test_batches), len(y_test_batches), len(x_train_batches), len(y_train_batches)
    # print x_test_batches[0]

    return x_train_batches, y_train_batches, x_test_batches, y_test_batches

def to_batch(arr, batch_size):
    return [np.array(arr[x*batch_size:(x+1)*batch_size]) for x in range(len(arr)/batch_size)]

train_X, train_Y = process_train("../data/spam_train.csv")
test_X = process_test("../data/spam_test.csv")
train_x_split, train_y_split = data_split_4_cross_valid(train_X, train_Y)
# get_batch_data(0)

