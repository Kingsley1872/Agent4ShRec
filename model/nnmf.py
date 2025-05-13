try:
    import tensorflow._api.v2.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import time
import numpy as np
import math
import pickle

class Model:
    def __init__(self, sess, dataset, is_text_init, is_text_pure, target_id, num_factor_1=100, num_factor_2=10, hidden_dimension=50,
                 learning_rate=0.001, reg_rate=0.01, epoch=600, batch_size=256,
                 show_time=False, T=5, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.dataset = dataset
        self.num_user = dataset.n_users
        self.num_item = dataset.n_items
        self.dataset.test_matrix_dok = self.dataset.test_matrix.todok()

        self.num_factor_1 = num_factor_1
        self.num_factor_2 = num_factor_2
        self.hidden_dimension = hidden_dimension
        self.show_time = show_time
        self.T = T
        self.display_step = display_step
        print("Model initialized.")

        self.dataset_train_matrix_coo = self.dataset.train_matrix.tocoo()
        self.user = self.dataset_train_matrix_coo.row.reshape(-1)
        self.item = self.dataset_train_matrix_coo.col.reshape(-1)
        self.rating = self.dataset_train_matrix_coo.data

        if is_text_init:
            if is_text_pure:
                with open('user_embeddings.pkl', 'rb') as user_file, open('item_embeddings.pkl', 'rb') as item_file:
                    self.init_user_embedding = pickle.load(user_file)
                    self.init_item_embedding = pickle.load(item_file)
            else:
                with open(f'user_embeddings_{target_id}.pkl', 'rb') as user_file, open(f'item_embeddings_{target_id}.pkl', 'rb') as item_file:
                    self.init_user_embedding = pickle.load(user_file)
                    self.init_item_embedding = pickle.load(item_file)
            self.init_user_embedding = {int(k): v for k, v in self.init_user_embedding.items()}
            self.init_item_embedding = {int(k): v for k, v in self.init_item_embedding.items()}

        self._build_network(is_text_init)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def _build_network(self, is_text_init):
        print("num_factor_1=%d, num_factor_2=%d, hidden_dimension=%d" % (
            self.num_factor_1, self.num_factor_2, self.hidden_dimension))
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder("float", [None], 'rating')

        if is_text_init:
            P_init = np.zeros((self.num_user, self.num_factor_1), dtype=np.float32)
            for user_id, embedding in sorted(self.init_user_embedding.items()):
                P_init[user_id] = embedding
            Q_init = np.zeros((self.num_item, self.num_factor_1), dtype=np.float32)
            for item_id, embedding in sorted(self.init_item_embedding.items()):
                Q_init[item_id] = embedding
        else:
            P_init = None
            Q_init = None
    
        if P_init is not None:
            P = tf.Variable(P_init, dtype=tf.float32, name="P")
        else:
            P = tf.Variable(tf.random_normal([self.num_user, self.num_factor_1], stddev=0.01), name="P")

        if Q_init is not None:
            Q = tf.Variable(Q_init, dtype=tf.float32, name="Q")
        else:
            Q = tf.Variable(tf.random_normal([self.num_item, self.num_factor_1], stddev=0.01), name="Q")

        U = tf.Variable(tf.random_normal([self.num_user, self.num_factor_2], stddev=0.01), name="U")
        V = tf.Variable(tf.random_normal([self.num_item, self.num_factor_2], stddev=0.01), name="V")

        input = tf.concat(values=[tf.nn.embedding_lookup(P, self.user_id),
                                  tf.nn.embedding_lookup(Q, self.item_id),
                                  tf.multiply(tf.nn.embedding_lookup(U, self.user_id),
                                              tf.nn.embedding_lookup(V, self.item_id))
                                  ], axis=1)

        regularizer = tf.keras.regularizers.l2(self.reg_rate)
        layer_1 = tf.layers.dense(inputs=input, units=2 * self.num_factor_1 + self.num_factor_2,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer, activation=tf.sigmoid,
                                  kernel_regularizer=regularizer)
        layer_2 = tf.layers.dense(inputs=layer_1, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_3 = tf.layers.dense(inputs=layer_2, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        layer_4 = tf.layers.dense(inputs=layer_3, units=self.hidden_dimension, activation=tf.sigmoid,
                                  bias_initializer=tf.random_normal_initializer,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=regularizer)
        output = tf.layers.dense(inputs=layer_4, units=1, activation=None,
                                 bias_initializer=tf.random_normal_initializer,
                                 kernel_initializer=tf.random_normal_initializer,
                                 kernel_regularizer=regularizer)
        self.pred_rating = tf.reshape(output, [-1])
        self.loss = tf.reduce_sum(tf.square(self.y - self.pred_rating)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
                            tf.norm(U) + tf.norm(V) + tf.norm(P) + tf.norm(Q))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        self.num_training = len(self.rating)
        total_batch = int(self.num_training / self.batch_size)
        idxs = np.random.permutation(self.num_training)
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        rating_random = list(self.rating[idxs])
        for i in range(total_batch):
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_rating = rating_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.user_id: batch_user,
                                                                            self.item_id: batch_item,
                                                                            self.y: batch_rating
                                                                            })
        return loss

    def test(self, test_data):
        error = 0
        error_mae = 0
        test_set = list(test_data.keys())
        for (u, i) in test_set:
            pred_rating_test = self.predict([u], [i])[0]
            error += (float(test_data.get((u, i))) - pred_rating_test) ** 2
            error_mae += (np.abs(float(test_data.get((u, i))) - pred_rating_test))
        rmse = np.sqrt(error / len(test_set))
        mae = error_mae / len(test_set)
        return rmse, mae

    def execute(self):
        loss_prev = float("inf")
        for epoch in range(self.epochs):
            loss_cur = self.train()
            if epoch % self.T == 0:
                print("epoch:\t", epoch, "\tloss:\t", loss_cur)
            loss_prev = loss_cur
        rmse, mae = self.test(self.dataset.test_matrix_dok)
        print("training done\tRMSE : ", rmse, "\tMAE : ", mae)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore(self, path):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def predict(self, user_id, item_id):
        if type(item_id) != list:
            item_id = [item_id]
        if type(user_id) != list:
            user_id = [user_id] * len(item_id)
        return self.sess.run([self.pred_rating], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]
