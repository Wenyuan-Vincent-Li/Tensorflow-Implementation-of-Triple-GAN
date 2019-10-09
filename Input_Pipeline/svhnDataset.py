## import module
import os, sys
import tensorflow as tf
import scipy.io
if os.getcwd().startswith("Path_Semi_GAN"):
    root_dir = os.getcwd()
else:
    root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)

class svhnDataset(object):
    """
    cedars dataset with label{0: no cancer 1: cancer in the image}
    """
    def __init__(self, data_dir, config, num_label = None, subset='train', use_augmentation = False):
        self.data_dir = os.path.join(data_dir, "Tfrecord")
        self.subset = subset
        self.use_augmentation = use_augmentation
        self.config = config
        self.num_label = num_label
        self.train_size = 73257

    def get_filenames(self):
        assert self.subset in ['train', 'test'], 'Invalid data subset "%s"' %self.subset
        if self.subset == 'train':
            return [os.path.join(self.data_dir, 'svhn_%s_%s.tfrecords'%(self.subset, str(self.num_label).zfill(6))),
                    os.path.join(self.data_dir, 'svhn_%s_%s.tfrecords'%(self.subset, \
                                                                        str(self.train_size - self.num_label).zfill(6)))]

        else:
            return [os.path.join(self.data_dir, 'svhn_%s.tfrecords'%self.subset)]

    def input_from_tfrecord_filename(self):
        ## Read in datasets according to filename and return a object list
        filename = self.get_filenames()
        if self.subset == 'test':
            dataset = []
            for name in filename:
                dataset.append(tf.data.TFRecordDataset(name))
        else:
            dataset = [tf.data.TFRecordDataset(filename[0]), tf.data.TFRecordDataset(filename[0]), tf.data.TFRecordDataset(filename[1])]
        return dataset

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64)
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        image = tf.cast(tf.reshape(image, [height, width, 3]), tf.float32)
        # image.set_shape([self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, 3])

        # map image to [-1, 1]
        image = image / 255 * 2 - 1  ## map pixel value to {-1, 1}
        # one-hot encoding for the label
        label = tf.one_hot(label, depth = self.config.NUM_CLASSES)

        ## pre-processing data do augmentation
        if self.use_augmentation:
            image, label = self.pre_processing(image, label)
        return image, label

    def pre_processing(self, image, label):
        pass
        return image, label

    def shuffle_and_repeat(self, dataset, repeat = 1):
        dataset = dataset.shuffle(buffer_size = \
                                      self.config.MIN_QUEUE_EXAMPLES + \
                                      30 * self.config.BATCH_SIZE, \
                                  )
        dataset = dataset.repeat(count = repeat)
        return dataset

    def batch(self, dataset, batch_size):
        dataset = dataset.batch(batch_size = batch_size)
        dataset = dataset.prefetch(buffer_size = batch_size * 5)
        return dataset

    def inputpipline_train_val(self, other):
        """
        Inputpipline that used for training
        :param other: validation dataset
        :return: init_op_train (list); init_op_val (list); lab_input (tensor);
                lab_output (tensor); train_unl_input (tensor)
        """
        # 1 Read in tfrecords
        dataset_train_lab_d, dataset_train_lab_c, dataset_train_unl = self.input_from_tfrecord_filename()
        dataset_val = other.input_from_tfrecord_filename()[0]
        # 2 Parser tfrecords and preprocessing the data
        dataset_train_lab_d = dataset_train_lab_d.map(self.parser, \
                                                  num_parallel_calls = self.config.BATCH_SIZE_L_D)
        dataset_train_lab_c = dataset_train_lab_c.map(self.parser, \
                                                      num_parallel_calls=self.config.BATCH_SIZE)
        dataset_train_unl = dataset_train_unl.map(self.parser, \
                                                  num_parallel_calls = self.config.BATCH_SIZE + self.config.BATCH_SIZE_U_D)
        dataset_val = dataset_val.map(self.parser, \
                                      num_parallel_calls = self.config.BATCH_SIZE)
        # 3 Shuffle and repeat
        dataset_train_lab_d = self.shuffle_and_repeat(dataset_train_lab_d, repeat = -1)
        dataset_train_lab_c = self.shuffle_and_repeat(dataset_train_lab_c, repeat=-1)
        dataset_train_unl = self.shuffle_and_repeat(dataset_train_unl, repeat = self.config.REPEAT)
        dataset_val = self.shuffle_and_repeat(dataset_val, repeat = 1)
        # 4 Batch it up
        dataset_train_lab_d = self.batch(dataset_train_lab_d, self.config.BATCH_SIZE_L_D)
        dataset_train_lab_c = self.batch(dataset_train_lab_c, self.config.BATCH_SIZE)
        dataset_train_unl = self.batch(dataset_train_unl, self.config.BATCH_SIZE + self.config.BATCH_SIZE_U_D)
        dataset_val = self.batch(dataset_val, self.config.BATCH_SIZE)
        # 5 Make iterator
        # first make the labeled data structure iterator
        lab_c_iterator = tf.data.Iterator.from_structure(dataset_train_lab_c.output_types, \
                                                       dataset_train_lab_c.output_shapes)
        lab_c_input, lab_c_output = lab_c_iterator.get_next()

        lab_d_iterator = dataset_train_lab_d.make_initializable_iterator()
        lab_d_input, lab_d_output = lab_d_iterator.get_next()

        # then make the unl data iterator
        train_unl_iterator = dataset_train_unl.make_initializable_iterator()
        train_unl_input, _ = train_unl_iterator.get_next()

        # finally make initializer
        init_op_train_lab_c = lab_c_iterator.make_initializer(dataset_train_lab_c)
        init_op_val_lab = lab_c_iterator.make_initializer(dataset_val)
        init_op_train = [init_op_train_lab_c, lab_d_iterator.initializer, train_unl_iterator.initializer]
        init_op_val = [init_op_val_lab]

        return init_op_train, init_op_val, [lab_c_input, lab_c_output, lab_d_input, lab_d_output, train_unl_input]


    def inputpipline_train_val_BS(self, other):
        """
        Inputpipline that used for training
        :param other: validation dataset
        :return: init_op_train (list); init_op_val (list); lab_input (tensor);
                lab_output (tensor); train_unl_input (tensor)
        """
        # 1 Read in tfrecords
        dataset_train_lab_d, dataset_train_lab_c, dataset_train_unl = self.input_from_tfrecord_filename()
        dataset_val = other.input_from_tfrecord_filename()[0]
        # 2 Parser tfrecords and preprocessing the data
        dataset_train_lab_d = dataset_train_lab_d.map(self.parser, \
                                                  num_parallel_calls = 20) ## 20
        dataset_train_lab_c = dataset_train_lab_c.map(self.parser, \
                                                      num_parallel_calls = 20) ## 50
        dataset_train_unl = dataset_train_unl.map(self.parser, \
                                                  num_parallel_calls = 20)
        dataset_val = dataset_val.map(self.parser, \
                                      num_parallel_calls = 20)
        # 3 Shuffle and repeat
        dataset_train_lab_d = self.shuffle_and_repeat(dataset_train_lab_d, repeat = -1)
        dataset_train_lab_c = self.shuffle_and_repeat(dataset_train_lab_c, repeat=-1)
        dataset_train_unl = self.shuffle_and_repeat(dataset_train_unl, repeat = self.config.REPEAT)
        dataset_val = self.shuffle_and_repeat(dataset_val, repeat = 1)
        # 4 Batch it up
        dataset_train_lab_d = self.batch(dataset_train_lab_d, self.config.BATCH_SIZE_L_D)
        dataset_train_lab_c = self.batch(dataset_train_lab_c, self.config.BATCH_SIZE_bG)
        dataset_train_unl = self.batch(dataset_train_unl, self.config.BATCH_SIZE_bG + self.config.BATCH_SIZE_gG \
                                       + self.config.BATCH_SIZE_gG - self.config.BATCH_SIZE_L_D)
        dataset_val = self.batch(dataset_val, self.config.BATCH_SIZE_bG)
        # 5 Make iterator
        # first make the labeled data structure iterator
        lab_c_iterator = tf.data.Iterator.from_structure(dataset_train_lab_c.output_types, \
                                                       dataset_train_lab_c.output_shapes)
        lab_c_input, lab_c_output = lab_c_iterator.get_next()

        lab_d_iterator = dataset_train_lab_d.make_initializable_iterator()
        lab_d_input, lab_d_output = lab_d_iterator.get_next()

        # then make the unl data iterator
        train_unl_iterator = dataset_train_unl.make_initializable_iterator()
        train_unl_input, _ = train_unl_iterator.get_next()

        # finally make initializer
        init_op_train_lab_c = lab_c_iterator.make_initializer(dataset_train_lab_c)
        init_op_val_lab = lab_c_iterator.make_initializer(dataset_val)
        init_op_train = [init_op_train_lab_c, lab_d_iterator.initializer, train_unl_iterator.initializer]
        init_op_val = [init_op_val_lab]

        return init_op_train, init_op_val, [lab_c_input, lab_c_output, lab_d_input, lab_d_output, train_unl_input]


    def inputpipline_testSet(self):
        """
        Inputpipline that used for validation or testing
        :return:
        """
        # 1 Read in tfrecords
        dataset_val = self.input_from_tfrecord_filename()[0]
        # 2 Parser tfrecords and preprocessing the data
        dataset_val = dataset_val.map(self.parser, \
                                      num_parallel_calls=self.config.BATCH_SIZE)
        # 3 Shuffle and repeat
        dataset_val = self.shuffle_and_repeat(dataset_val, repeat = 1)
        # 4 Batch it up
        dataset_val = self.batch(dataset_val, self.config.BATCH_SIZE)
        # 5 Make iterator
        iterator = dataset_val.make_initializable_iterator()
        init_op = iterator.initializer
        image_batch, label_batch = iterator.get_next()
        image_batch.set_shape([self.config.BATCH_SIZE] + self.config.IMAGE_DIM)
        label_batch.set_shape([self.config.BATCH_SIZE, self.config.NUM_CLASSES])
        return image_batch, label_batch, init_op

    def inputpipline_customizedSet(self, filename_list):
        """
        Inputpipline for a customized tfrecord file
        :param filename:
        :return:
        """
        # 1 Read in tfrecords
        filename_list = [os.path.join(self.data_dir, x) for x in filename_list]
        dataset = tf.data.TFRecordDataset(filename_list)
        # 2 Parser tfrecords and preprocessing the data
        dataset = dataset.map(self.parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)
        # 3 Shuffle and repeat
        dataset = self.shuffle_and_repeat(dataset, repeat=self.config.REPEAT)
        # 4 Batch it up
        dataset = self.batch(dataset, self.config.BATCH_SIZE)

        # 5 Make iterator
        iterator = dataset.make_initializable_iterator()
        init_op = iterator.initializer
        image_batch, label_batch = iterator.get_next()
        image_batch.set_shape([self.config.BATCH_SIZE] + self.config.IMAGE_DIM)
        label_batch.set_shape([self.config.BATCH_SIZE, self.config.NUM_CLASSES])
        return image_batch, label_batch, [init_op]


def test_main_train_val():
    from config import Config

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs

    class Temp_Config(Config):
        BATCH_SIZE = 200
        VAL_STEP = 1
        REPEAT = -1
        NUM_CLASSES = 10

    tmp_config = Temp_Config()
    data_dir = os.path.join(root_dir, "DataSet/svhn")

    num_label = 2000
    train_data = svhnDataset(data_dir, tmp_config, num_label, 'train', True)
    val_data = svhnDataset(data_dir, tmp_config, None, 'test', False)

    init_op_train, init_op_val, NNIO = train_data.inputpipline_train_val(val_data)
    x_l_c, y_l_c, x_l_d, y_l_d, x_u = NNIO


    with tf.Session() as sess:
        sess.run(init_op_train)
        x_l_c_o, y_l_c_o, x_l_d_o, y_l_d_o, x_u_o = sess.run([x_l_c, y_l_c, x_l_d, y_l_d, x_u])
        sess.run(init_op_val)
        for _ in range(tmp_config.VAL_STEP):
            lab_input_o, lab_output_o = \
                sess.run([x_l_c, y_l_c])
            val_batch_shape = lab_input_o.shape

    ## Print the Statistics
    print("TRAN_VAL STATISTICS: \n", \
          "X_L_C: {}.\n".format(x_l_c_o.shape), \
          "Y_L_C: {} \n".format(y_l_c_o.shape), \
          "X_L_D: {}.\n".format(x_l_d_o.shape), \
          "Y_L_D: {} \n".format(y_l_d_o.shape), \
          "X_U: {} \n".format(x_u_o.shape), \
          "BATCH_SIZE_VAL:", val_batch_shape)

def test_val():
    from config import Config

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs

    class Temp_Config(Config):
        BATCH_SIZE = 1
        REPEAT = 1
        NUN_CLASSES = 10

    tmp_config = Temp_Config()
    data_dir = os.path.join(root_dir, "DataSet/svhn", None, 'test')

    val_data = svhnDataset(data_dir, tmp_config)

    lab_input, lab_output, init_op = val_data.inputpipline_testSet()

    num_image = 0
    num_batch = 0
    with tf.Session() as sess:
        sess.run(init_op)
        while True:
            try:
                lab_input_o, lab_output_o = \
                    sess.run([lab_input, lab_output])
                num_image += lab_input_o.shape[0]
                num_batch += 1
            except tf.errors.OutOfRangeError:
                batch_shape = lab_input_o.shape
                break;

    ## Print the Statistics
    print("VAL STATISTICS: \n", \
          "NUM_DATA: %d \n" % num_image, \
          "NUM_BATCH_PER_EPOCH: %d \n" % num_batch, \
          "BATCH_SIZE_TRAIN:", batch_shape)

def test_customized():
    from config import Config

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs

    class Temp_Config(Config):
        BATCH_SIZE = 1
        REPEAT = 1
        NUN_CLASSES = 10

    tmp_config = Temp_Config()
    data_dir = os.path.join(root_dir, "DataSet/svhn")

    data = svhnDataset(data_dir, tmp_config, None, 'test', False)
    filename_list = ["svhn_train_072257.tfrecords"]
    lab_input, lab_output, init_op = data.inputpipline_customizedSet(filename_list)

    num_image = 0
    num_batch = 0
    with tf.Session() as sess:
        sess.run(init_op)
        while True:
            try:
                lab_input_o, lab_output_o = \
                    sess.run([lab_input, lab_output])
                num_image += lab_input_o.shape[0]
                num_batch += 1
            except tf.errors.OutOfRangeError:
                batch_shape = lab_input_o.shape
                break;

    ## Print the Statistics
    print("DATA STATISTICS: \n", \
          "NUM_DATA: %d \n" % num_image, \
          "NUM_BATCH_PER_EPOCH: %d \n" % num_batch, \
          "BATCH_SIZE_DATA:", batch_shape)

if __name__ == "__main__":
    test_main_train_val()