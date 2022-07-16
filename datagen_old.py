from sklearn.model_selection import train_test_split

import tensorflow as tf

class DataGen:
    def __init__(self, image_dir_path):
        self.image_dir_path = image_dir_path
        self.image_names = None
        self.master_df = None

    def _parse_function(self, filename, age, gender):#convert file names to actual tensors(numpy arrays)
        image_string = tf.io.read_file(filename)
        image_decoded = tf.io.decode_jpeg(image_string, channels=1)    # channels=1 to convert to grayscale, channels=3 to convert to RGB.
        # image_resized = tf.image.resize(image_decoded, [200, 200])
        image_decoded = tf.cast(image_decoded, tf.float32)
        age = tf.one_hot(age, 7)
        gender = tf.one_hot(gender, 2)
        
        return image_decoded, age, gender

    def train_test_split(self, test_size=0.3, batch_size=512):
        X = self.master_df[['filename', 'target', 'gender']]
        y = self.master_df[['target', 'gender']]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        train_filenames_tensor = tf.constant(list(X_train['filename']))
        train_labels_tensor = tf.constant(list(y_train))
        test_filenames_tensor = tf.constant(list(X_test['filename']))
        test_labels_tensor = tf.constant(list(y_test))

        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames_tensor, train_labels_tensor))
        train_dataset = train_dataset.map(self._parse_function)
        #train_dataset = train_dataset.repeat(3)
        train_dataset = train_dataset.batch(batch_size)

        test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames_tensor, test_labels_tensor))
        test_dataset = test_dataset.map(self._parse_function)
        #test_dataset = test_dataset.repeat(3)
        test_dataset = test_dataset.batch(batch_size)

        return (train_dataset, test_dataset)

        #print([x.get_shape().as_list() for x in train_dataset._tensors])

        #x = np.array([np.array(Image.open(fname)) for fname in list(X_train['filename'])]) #returns with shape of (23440, 200, 200, 3)

    


    
