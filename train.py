import tensorflow as tf
from read_utils import TextConverter, batch_generator
from model import CharRNN
import os
import jieba
import pickle
import codecs

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('name', 'test', 'name of the model')                              # 'name': model存储目录
tf.flags.DEFINE_string('input_file','./data/test.txt','utf8 encoded text file')   # ./data/music.txt
tf.flags.DEFINE_string('converter_path','','./model/name/converter.pkl')   # ./model/novel/converter.pkl
tf.flags.DEFINE_string('checkpoint_path','','checkpoint_path')

tf.flags.DEFINE_integer('max_steps', 10000, 'max steps to train')
tf.flags.DEFINE_integer('save_every_n', 1000, 'save the model every n steps')  # 每隔多少次储存一次预训练模型
tf.flags.DEFINE_integer('log_every_n', 1000, 'log to the screen every n steps')
tf.flags.DEFINE_integer('max_vocab', 5000, 'max char number')
tf.flags.DEFINE_integer('num_seqs', 100, 'number of seqs in one batch')  # 文本长度必须大于 num_seqs * num_steps
tf.flags.DEFINE_integer('num_steps', 100, 'length of one seq')

tf.flags.DEFINE_integer('lstm_size', 128, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 2, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 128, 'size of embedding')
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning_rate')
tf.flags.DEFINE_float('train_keep_prob', 0.5, 'dropout rate during training')


def file_to_text(file):
    with open(file, "r", encoding='UTF-8') as f:
        text = []
        for line in f.readlines():
            line = line.replace("\n", "").strip()
            if line == "" or line is None:
                continue
            line = ' '.join(jieba.cut(line))
            for word in line.split(' '):
                if word !=  '\ufeff':
                    text.append(word)
    return text


def main(_):
    # 创建模型存储路径
    model_path = os.path.join('model', FLAGS.name)
    if os.path.exists(model_path) is False:
        os.makedirs(model_path)

    # 文本切成单个字(适合英语和诗)
    #with codecs.open(FLAGS.input_file, encoding='utf-8') as f:
        #text = f.read()

    # 文本按顺序分成词语，存成converter.pkl
    if FLAGS.input_file:
        text = file_to_text(FLAGS.input_file)
        pickle.dump(text,open(os.path.join(model_path,'text.pickle'),'wb'))
        converter = TextConverter(text, FLAGS.max_vocab)
        converter.save_to_file(os.path.join(model_path, 'converter.pkl'))
    # 如果词语已分好，直接读取
    if FLAGS.converter_path:
        text = pickle.load(open(os.path.join(model_path,'text.pickle'),'rb'))
        converter = TextConverter(filename=FLAGS.converter_path)

    # 词语转成词频
    arr = converter.text_to_arr(text)
    g = batch_generator(arr, FLAGS.num_seqs, FLAGS.num_steps)
    print(converter.vocab_size)

    model = CharRNN(converter.vocab_size,
                    num_seqs=FLAGS.num_seqs,
                    num_steps=FLAGS.num_steps,
                    lstm_size=FLAGS.lstm_size,
                    num_layers=FLAGS.num_layers,
                    learning_rate=FLAGS.learning_rate,
                    train_keep_prob=FLAGS.train_keep_prob,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size
                    )

    #model.load(FLAGS.checkpoint_path)

    model.train(g,
                FLAGS.max_steps,
                model_path,
                FLAGS.save_every_n,
                FLAGS.log_every_n,
                )


if __name__ == '__main__':
    tf.app.run()

