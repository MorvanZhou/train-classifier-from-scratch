import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

DOWNLOAD = False

# download data from : http://archive.ics.uci.edu/ml/datasets/Car+Evaluation
if DOWNLOAD:
    data_path, _ = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.csv")

# data processing
col_name = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
data = pd.read_csv("car.csv", names=col_name)


# covert class data to onehot
new_n_col = sum([len(pd.unique(data[name])) for name in col_name])
new_data = pd.get_dummies(data, prefix=col_name)


# prepare training data
new_data = new_data.values.astype(np.float32)       # change to numpy array and float32
np.random.shuffle(new_data)
sep = int(0.7*len(new_data))
train_data = new_data[:sep]                         # training data (70%)
test_data = new_data[sep:]                          # test data (30%)


# build network
tf_input = tf.placeholder(tf.float32, [None, 25], "input")
tfx = tf_input[:, :21]
tfy = tf_input[:, 21:]

l1 = tf.layers.dense(tfx, 128, tf.nn.relu, name="l1")
l2 = tf.layers.dense(l1, 128, tf.nn.relu, name="l2")
out = tf.layers.dense(l2, 4, name="l3")
prediction = tf.nn.softmax(out, name="pred")

loss = tf.losses.softmax_cross_entropy(onehot_labels=tfy, logits=out)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tfy, axis=1), predictions=tf.argmax(out, axis=1),)[1]
opt = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = opt.minimize(loss)

sess = tf.Session()
sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

# training
plt.ion()
for t in range(2000):
    batch_index = np.random.randint(len(train_data), size=32)
    sess.run(train_op, {tf_input: train_data[batch_index]})
    if t % 50 == 0:
        acc_, pred_ = sess.run([accuracy, prediction], {tf_input: test_data})
        print(
            "Step: %i" % t,
            "| Accurate: %.2f" % acc_,
        )

        # visualize training
        plt.cla()
        for c in range(4):
            plt.bar(x=c+0.1, height=sum((np.argmax(pred_, axis=1) == c)), width=0.2, color='red')
            plt.bar(x=c-0.1, height=sum((np.argmax(test_data[:, 21:], axis=1) == c)), width=0.2, color='blue')
        plt.xticks(range(4), ["acc", "good", "unacc", "vgood"])
        plt.pause(0.1)

plt.ioff()
plt.show()