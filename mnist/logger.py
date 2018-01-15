import numpy as np
import os
import datetime
import collections
import shutil
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import chainer

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

    use_tensorboard = True
except ImportError:
    print('tensorflow is not installed')
    use_tensorboard = False

out_dir = None

_since_beginning = collections.defaultdict(lambda: {})
_iter = [0]

# tensorboard
if use_tensorboard:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    tf_array = tf.placeholder(tf.float32)
    summaries = {}
    for x in ['train_accuracy', 'test_accuracy']:
        summaries[x] = tf.summary.scalar(x, tf_array)
    summary_writer = None


def init(args):
    global out_dir
    if args.resume is not None:
        out = args.resume
    else:
        out = datetime.datetime.now().strftime('%m%d%H') + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, args.log_dir, out))

    if os.path.exists(out_dir):
        ans = input('overwrite "{}" (y/n)'.format(out_dir))
        if ans == 'y' or ans == 'Y':
            print('move existing directory to dump')
            try:
                shutil.move(out_dir, out_dir.replace(args.log_dir, 'dump'))
            except:
                print(out_dir)
                shutil.rmtree(out_dir.replace(args.log_dir, 'dump'))
                shutil.move(out_dir, out_dir.replace(args.log_dir, 'dump'))
        else:
            print('try again')
            exit()
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'queries'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'plot'), exist_ok=True)

    # setting
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            print('{} = {}'.format(k, v))
            f.write('{} = {}\n'.format(k, v))

    if use_tensorboard:
        global summary_writer
        summary_dir = os.path.join(out_dir, "summaries")
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)


def plot(name, value):
    _since_beginning[name][_iter[0]] = value

    if use_tensorboard:
        if name in summaries:
            summary = sess.run(summaries[name], feed_dict={tf_array: value})
            summary_writer.add_summary(summary, _iter[0])


def save(model, query_X, query_y):
    np.save(os.path.join(out_dir, 'queries', 'query_images_{}'.format(_iter[0])), query_X)
    np.save(os.path.join(out_dir, 'queries', 'query_label_{}'.format(_iter[0])), query_y)
    chainer.serializers.save_npz(os.path.join(out_dir, 'models', 'cnn_{}.model'.format(_iter[0])), model)


def flush():
    log = ''
    log += "epoch {}\n".format(_iter[0])
    for name, vals in sorted(_since_beginning.items()):
        log += " {}\t{:.5f}\n".format(name, vals[_iter[0]])
        x_vals = np.sort(list(_since_beginning[name].keys()))
        y_vals = [_since_beginning[name][x] for x in x_vals]

        plt.clf()
        plt.plot(x_vals, y_vals)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.savefig(os.path.join(out_dir, 'plot', name.replace(' ', '_') + '.jpg'))

    print(log)
    with open(os.path.join(out_dir, 'log'), 'a+') as f:
        f.write(log + '\n')

    _iter[0] += 1
