from importlib import import_module
from absl import app, flags


flags.DEFINE_string('cmd', None, '')
flags.mark_flag_as_required('cmd')

flags.DEFINE_string('target', 'label', '')
flags.DEFINE_list('dense_feat', [f'I{i}' for i in range(1, 14)], '')
flags.DEFINE_list('sparse_feat', [f'C{i}' for i in range(1, 27)], '')

flags.DEFINE_integer('file_rows', 500000, '')
flags.DEFINE_list(
    'train_files', [f'./data/train_{i:0>2d}' for i in range(2)], '')
flags.DEFINE_list(
    'valid_files', [f'./data/train_{i:0>2d}' for i in range(80, 81)], '')

flags.DEFINE_bool('continue_train', False, '')
flags.DEFINE_integer('random_seed', 2020, '')
flags.DEFINE_integer('batch_size', 50, '')
flags.DEFINE_integer('epochs', 1, '')
flags.DEFINE_integer('early_stop', 1, '')


def main(argv):
    import_module(flags.FLAGS.cmd).main()


if __name__ == '__main__':
    app.run(main)
