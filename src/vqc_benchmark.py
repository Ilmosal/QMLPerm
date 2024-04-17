import numpy as np
import json
from vqc.tsp import solve
from vqc.data_perm import WineDataset, Dataset, BalancedILPDDataset, ILPDDataset, DiabetesDataset, HeartFailureDataset, Fertility, TransfusionDataset, TeachingAssistantDataset, HayesRothDataset, BanknoteDataset, AcuteInflammationsDataset, ImmunotherapyDataset
from data_reuploading import DataReuploadingClassifier
from quanvolutional_neural_network import QuanvolutionalNeuralNetwork
num_qubits = 4
from bars_and_stripes import generate_bars_and_stripes

# with open("./src/two_curves_data.json", "r") as read_file:
    # data = json.load(read_file)
with open("./src/bars_and_stripes.json", "r") as read_file:
    data = json.load(read_file)

train_data = np.asarray(data["data"])
test_data = np.asarray(data["data"])
train_labels = np.asarray(data["labels"])
test_labels = np.asarray(data["labels"])


rand_perm = np.arange(num_qubits)
np.random.shuffle(rand_perm)

# var_mat = np.abs(np.corrcoef(train_data[:,:], rowvar=False))
# max_perm, dist = solve(var_mat, maximize = True, linear = False)
# min_perm, dist = solve(var_mat, linear = False)

# # perm_train_data = train_data[:, rand_perm]
# perm_train_data = train_data[:, max_perm]

# vqc = DataReuploadingClassifier(n_layers=num_qubits, observable_type='full', convergence_interval=600, max_steps=10000)
# vqc.fit(train_data, train_labels)
# preds = vqc.predict(test_data)
# labels = test_labels

# acc = 0

# for l, p in zip(labels, preds):
#     if abs(l-p) < 1e-5:
#         acc += 1

# print(f"Data reup acc: {acc / len(labels)}")

qnn = QuanvolutionalNeuralNetwork(qkernel_shape=num_qubits, kernel_shape=num_qubits + 1, convergence_interval=200)
# train_labels.shape = (len(train_labels), 1)
train_data = train_data.reshape(len(train_data), len(train_data[0][0]) ** 2)
test_data = test_data.reshape(len(test_data), len(test_data[0][0]) ** 2)
qnn.fit(train_data, train_labels)
preds = qnn.predict(test_data)
labels = test_labels

acc = 0

for l, p in zip(labels, preds):
    if abs(l-p) < 1e-5:
        acc += 1

print(f"QNN acc: {acc / len(labels)}")


# class NumpyArrayEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)
# X, y = generate_bars_and_stripes(n_samples=200, width=8, height= 8, noise_std=0.1)
# print(y)
# f = open('bars_and_stripe.json', 'w')
# data = {
#     "data": X,
#     "labels": y
# }
# json.dump(data, f, indent=4, cls=NumpyArrayEncoder)
# f.close()