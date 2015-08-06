import math
import copy
import numpy as np
import util

# calculate the output of activation functions in every layer
# save the result in a list: values
# values[0] is the input
def update_values(x, paras):
    ws, bs = paras
    values = [x]
    for layer in range(len(ws)-1):
        values.append(vec_tanh(ws[layer].dot(values[layer]) + bs[layer]))
    y = vec_exp(ws[-1].dot(values[-1]) + bs[-1])
    values.append(y/sum(y))
    return values

# initialize weights and biases to be zeros
def initialize_paras_zero(ns):
    ws, bs = [], []
    for layer in range(len(ns)-1):
        ws.append(np.zeros((ns[layer+1], ns[layer])))
        bs.append(np.zeros((ns[layer+1], 1)))
    return ws, bs

# initialize weights and biases randomly
# range is [-offset*scale, (1-offset)*scale]
# in this project, we set offset to be 0.5 and scale to be 0.4
# therefore range is [-0.2, 0.2]
def initialize_paras_random(ns, offset, scale):
    ws, bs = [], []
    for layer in range(len(ns)-1):
        w = (np.random.random_sample((ns[layer+1], ns[layer])) - offset)*scale
        b = (np.random.random_sample((ns[layer+1], 1        )) - offset)*scale
        ws.append(w)
        bs.append(b)
    return ws, bs

# use backpropagation to calculate the gradients
# add the gradients to delta_paras
# data_paras consist of dalta_ws and delta_bs
# which stores the weights and biases matrices, respectively
def backprop(paras, values, target, tau, delta_paras):
    ws, bs = paras
    delta_ws, delta_bs = delta_paras

    delta = (target - values[-1]) * tau

    for layer in range(len(ws)-1, 0, -1): # len(ws)-1, ... 2, 1
        delta_bs[layer] += delta
        delta_ws[layer] += delta.dot(values[layer].T)
        delta = ws[layer].T.dot(delta) * (1 - values[layer]**2)

    delta_bs[0] += delta
    delta_ws[0] += delta.dot(values[0].T)

# add the delta_paras to paras
def update_paras(paras, delta_paras):
    ws, bs = paras
    delta_ws, delta_bs = delta_paras
    for layer in range(len(ws)):
        ws[layer] += delta_ws[layer]
        bs[layer] += delta_bs[layer]

# predict label for x
def predict_label(x, paras):
    ws, bs = paras
    value = x
    for layer in range(len(ws)-1):
        value = vec_tanh(ws[layer].dot(value) + bs[layer])
    return (ws[-1].dot(value) + bs[-1]).argmax()

# return the accuracy calculated on traing set
def accuracy_training_set(paras, instances_by_label):
    count = 0
    for label in range(10):
        for instance in instances_by_label[label]:
            if predict_label(instance, paras) == label:
                count += 1
    return count

# return the accuracy calculated on test set
def accuracy_test_set(paras, test_instances, test_labels):
    count = 0
    for (index, instance) in enumerate(test_instances):
        if predict_label(instance, paras) == test_labels[index]:
            count += 1
    return count

# training algorithm for neural network
# input: structure (a list of numbers of nodes in each layer)
#        instances_by_label (a dictionary {label : [instance]}
#                            key is label, value is a list of instances)
#        test_instances (a list of instances in validation set)
#        tset_labels    (a list of labels of test_instances)
# return: max_accuracy (accuracies on training set and validation set)
#         paras (a tuple of ws and bs
#                which stores weights and biases matrices for each layer)
def train_nn(structure, instances_by_label, test_instances, test_labels):
    tau = 0.01
    momentum = 0.5
    total_iterations = 60000
    gamma = 0.00001

    paras = initialize_paras_random(structure, 0.5, 0.4)
    delta_paras = initialize_paras_zero(structure)

    max_accuracy = (0, 0)
    max_paras = None

    last_accuracy = 0
    last_status = None
    
    k = 0
    while k < total_iterations and tau > 0.0001:
        pre_delta_paras = delta_paras
        delta_paras = initialize_paras_zero(structure)
        for label in range(10):
            instances = instances_by_label[label]
            index = np.random.randint(0, len(instances)-1)
            
            target = np.zeros((structure[-1], 1))
            target[label][0] = 1
            
            values = update_values(instances[index], paras)
            backprop(paras, values, target, tau, delta_paras)
                
        ws, bs = paras
        delta_ws, delta_bs = delta_paras
        pre_delta_ws, pre_delta_bs = pre_delta_paras
        for layer in range(len(delta_ws)):
            delta_ws[layer] -= ws[layer] * gamma
            delta_bs[layer] -= bs[layer] * gamma
            delta_ws[layer] += pre_delta_ws[layer] * momentum
            delta_bs[layer] += pre_delta_bs[layer] * momentum
        update_paras(paras, delta_paras)
                
        if k%2000 == 1999:
            accuracy = accuracy_training_set(paras, instances_by_label)
            test_accuracy = accuracy_test_set(paras, test_instances, test_labels)
            print "%d\t%d\t%d" % (k, test_accuracy, accuracy)

            if accuracy < last_accuracy:
                paras, delta_paras = copy.deepcopy(last_status)
                accuracy = last_accuracy
                tau /= 2
            else:
                last_accuracy = accuracy
                last_status = copy.deepcopy( (paras, delta_paras) )

            if accuracy > max_accuracy[0]:
                max_accuracy = (accuracy, test_accuracy)
                max_paras = copy.deepcopy(paras)
        k += 1

    return max_accuracy, paras

# a wrapper function of train_nn which is used for stage2
def get_accuracy(structure):
    structure = [25] + list(structure) + [10]
    instances_by_label = util.read_training("train.csv")
    for label in instances_by_label:
        for instance in instances_by_label[label]:
            instance /= max(instance.max(), -instance.min())
    
    np.random.seed(0)
    test_instances = []
    test_labels = []
    for label in instances_by_label:
        np.random.shuffle(instances_by_label[label])
        test_instances.extend(instances_by_label[label][:1000])
        test_labels.extend([label]*1000)
        instances_by_label[label] = instances_by_label[label][1000:]

    max_accuracy, paras = train_nn(structure, instances_by_label, test_instances, test_labels)
    return max_accuracy[1]

# definitons of vector-version tanh and exp
vec_tanh = np.vectorize(math.tanh)
vec_exp  = np.vectorize(math.exp)

if __name__ == "__main__":
    structure = [25, 100, 10]

    instances_by_label = util.read_training("train.csv")

    # normalize so that tha absolute value in an input doesn't exceed 1
    for label in instances_by_label:
        for instance in instances_by_label[label]:
            instance /= max(instance.max(), -instance.min())
    
    np.random.seed(0)
    test_instances = []
    test_labels = []
    for label in instances_by_label:
        np.random.shuffle(instances_by_label[label])
        test_instances.extend(instances_by_label[label][:1000])
        test_labels.extend([label]*1000)
        instances_by_label[label] = instances_by_label[label][1000:]

    max_accuracy, paras = train_nn(structure, instances_by_label, test_instances, test_labels)
