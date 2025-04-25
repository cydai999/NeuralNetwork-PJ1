import os
import time
from struct import unpack

import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import mynn as nn

# set seed
np.random.seed(123)

# load data
# load train and valid set
train_images_path = r'.\dataset\MNIST\train-images.idx3-ubyte'
train_labels_path = r'.\dataset\MNIST\train-labels.idx1-ubyte'

with open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# randomly select a 0.1x sample
idx = np.random.permutation(train_imgs.shape[0])
train_data, train_labels = train_imgs[idx], train_labs[idx]
valid_data, valid_labels = train_data[:2000], train_labels[:2000]
train_data, train_labels = train_data[10000: 20000], train_labels[10000: 20000]

# normalize
train_data = train_data / train_data.max()
valid_data = valid_data / valid_data.max()

valid_set = [valid_data, valid_labels]
train_set = [train_data, train_labels]

# load test set
test_images_path = r'.\dataset\MNIST\t10k-images.idx3-ubyte'
test_labels_path = r'.\dataset\MNIST\t10k-labels.idx1-ubyte'

with open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28 * 28)

with open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# normalize
test_data = test_imgs / test_imgs.max()

test_set = [test_data, test_labs]

# set params
# init_lrs = [1e-1, 1e-2, 1e-3]
# hidden_sizes = [2000, 1000, 500]
# weight_decay_params = [1e-2, 1e-3, 1e-4]
init_lrs = [1e-1]
hidden_sizes = [1000]
weight_decay_params = [1e-3]

# init model
act_func = 'LeakyReLU'
step_size = 5
batch_size = 32
# gammas = [1]
gammas = [1, 0.5, 0.1, 0.05]

for init_lr in init_lrs:
    for hidden_size in hidden_sizes:
        for weight_decay_param in weight_decay_params:
            for gamma in gammas:
                size_list = [784, hidden_size, 10]
                weight_decay_list = [weight_decay_param, weight_decay_param]

                model = nn.models.MLPModel(size_list=size_list, act_func=act_func, weight_decay_list=weight_decay_list)
                # optimizer = nn.optimizers.SGD(model, init_lr=init_lr)
                optimizer = nn.optimizers.SGDMomentum(model, init_lr=init_lr)
                lr_scheduler = nn.lr_schedulers.StepLR(optimizer, step_size=step_size, gamma=gamma)
                metric = nn.metric.accuracy
                loss_fn = nn.loss_fn.CrossEntropyLoss(model)
                runner = nn.runner.Runner(model, loss_fn, metric, batch_size=batch_size, optimizer=optimizer,
                                          lr_scheduler=lr_scheduler)

                # train
                epoch = 20
                save_dir = f'./saved_models/MLP/{init_lr}-{hidden_size}-{weight_decay_param}-{gamma}'
                # save_dir = f'./saved_models/MLP/SGD'
                log_iter = 100
                print(f'init_lr:{init_lr}\n'
                      f'hidden_size:{hidden_size}\n'
                      f'weight_decay_param:{weight_decay_param}\n'
                      f'gamma:{gamma}')
                print('[Train]Begin training...')
                start_time = time.time()
                runner.train(train_set, valid_set, epoch, save_dir, log_iter)
                training_time = time.time() - start_time
                print(f'[Train]Training completed! Training time:{training_time}')

                # test
                test_data, test_labels = test_set
                print('[Test]Begin testing...')
                loss, score = runner.eval(test_data, test_labels)
                print('[Test]Testing completed!')
                print(f'[Test]loss:{loss}, score:{score}')

                # save result
                result_path = os.path.join(save_dir, 'testing_result.json')
                result = {'loss': loss, 'score': score, 'training_time': training_time}
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f)

                # plot
                fg, axis = plt.subplots(1, 2)
                axis[0].plot(runner.train_loss, label='train_loss', color='r')
                axis[0].plot(runner.valid_loss, label='valid_loss', color='b')
                axis[0].set_xlabel('iteration')
                axis[0].set_ylabel('loss')
                axis[0].legend(loc='upper right')

                axis[1].plot(runner.train_score, label='train_score', color='r')
                axis[1].plot(runner.valid_score, label='valid_score', color='b')
                axis[1].set_xlabel('iteration')
                axis[1].set_ylabel('score')
                axis[1].legend(loc='upper right')

                fg.tight_layout()
                plt.savefig(os.path.join(save_dir, 'loss_accuracy_plot.png'))

                # save_params
                params = {'model': {
                    'size_list': size_list,
                    'act_func': act_func,
                    'weight_decay_list': weight_decay_list
                },
                    'optimizer': {
                        'type': 'Momentum',
                        'init_lr': init_lr
                    },
                    'lr_scheduler': {
                        'type': 'stepLR',
                        'step_size': step_size,
                        'gamma': gamma
                    },
                    'metric': {
                        'type': 'accuracy'
                    },
                    'loss function': {
                        'type': 'cross entropy'
                    },
                    'train': {
                        'batch_size': batch_size,
                        'epoch': epoch,
                        'log_iter': log_iter
                    }
                }

                json_path = os.path.join(save_dir, 'params.json')

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(params, f)



