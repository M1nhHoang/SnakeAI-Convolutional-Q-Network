import tensorflow as tf
from tensorflow.keras import layers, models
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Giảm bộ nhớ được chia sẻ cho GPU
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])


class Linear_QNet(tf.keras.Model):
    def __init__(self, input_img_size, output_size):
        super(Linear_QNet, self).__init__()
        self.input_img_size = input_img_size
        # Define the CNN layers for image processing
        self.cnn_conv1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_img_size)
        self.cnn_maxpool1 = layers.MaxPooling2D((2, 2))
        self.cnn_conv2 = layers.Conv2D(64, (3, 3), activation='relu')
        self.cnn_maxpool2 = layers.MaxPooling2D((2, 2))
        self.cnn_flatten = layers.Flatten()

        # Define the Dense layers for the vector input
        self.dense_vector = layers.Dense(64, activation='relu')

        # Define the Dense layers for the combined features
        self.dense_combined1 = layers.Dense(128, activation='relu')
        self.dense_combined2 = layers.Dense(64, activation='relu')

        # Output layer
        self.output_layer = layers.Dense(output_size, activation='softmax')

    def call(self, inputs):
        image_input, vector_input = inputs

        # Expand dimensions to add a batch_size
        # image_input = tf.cast(tf.expand_dims(image_input, axis=0), dtype=tf.float32)

        # Process the image input through CNN
        x_image = self.cnn_conv1(image_input)
        x_image = self.cnn_maxpool1(x_image)
        x_image = self.cnn_conv2(x_image)
        x_image = self.cnn_maxpool2(x_image)
        x_image = self.cnn_flatten(x_image)

        # Process the vector input through Dense layers
        x_vector = self.dense_vector(vector_input)

        # Concatenate the features
        combined_features = layers.Concatenate()([x_image, x_vector])

        # Process the combined features through Dense layers
        x_combined = self.dense_combined1(combined_features)
        x_combined = self.dense_combined2(x_combined)

        # Output layer
        output = self.output_layer(x_combined)

        return output

    def save(self, file_name='model.h5'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        self.save_weights(file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.loss_fn = tf.losses.MeanSquaredError()

    def train_step(self, state, action, reward, next_state, done):
        # print(state)
        # state = tf.convert_to_tensor(state, dtype=tf.float32)
        # next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.int64)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        try:
            len(done)
        except:
            # state = tf.expand_dims(state, 0)
            # next_state = tf.expand_dims(next_state, 0)
            state, next_state = [state], [next_state]
            action = tf.expand_dims(action, 0)
            reward = tf.expand_dims(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        with tf.GradientTape() as tape:
            state = tf.data.Dataset.from_generator(lambda: state, output_signature=(tf.TensorSpec(shape=(None, 200, 150, 3),
                                                                                                  dtype=tf.float32), tf.TensorSpec(shape=(None, 11), dtype=tf.float32)))
            next_state = tf.data.Dataset.from_generator(lambda: next_state, output_signature=(tf.TensorSpec(shape=(None, 200, 150, 3),
                                                                                                            dtype=tf.float32), tf.TensorSpec(shape=(None, 11), dtype=tf.float32)))

            pred = self.model(state, training=True)

            target = tf.identity(pred)
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:
                    Q_new = reward[idx] + self.gamma * tf.reduce_max(self.model(next_state))

                print("target", target)
                print("Q_new", Q_new)
                target = tf.tensor_scatter_nd_update(target, indices=[[idx, tf.argmax(action[idx]).numpy()]], updates=[Q_new])

            # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
            # pred.clone()
            # preds[argmax(action)] = Q_new
            loss = self.loss_fn(target, pred)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
