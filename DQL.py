from QL import QN
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

class DQN(QN) :

    def build(self) :
        self.set_optimizer_loss()
        self.build_model()
        self.target_model = keras.models.clone_model(self.model)
        self.update_target()

    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state

    def update_target(self) :
        self.target_model.set_weights(self.model.get_weights())

    def get_best_action(self, state) :
        action_values = self.model(state)
        return np.argmax(action_values), action_values

    def set_input_to_output(self, inputs) :
        """
        get inputs and return outputs
        """
        raise NotImplementedError

    def build_model(self) :
        state_shape = list(self.env.observation_space.shape)
        inputs = keras.Input(
            dtype = tf.uint8, 
            shape = (
                state_shape[0],
                state_shape[1],
                state_shape[2] * self.config.state_history,
            ),
            # batch_size = self.config.batch_size,
        )
        state = self.process_state(inputs)
        outputs = self.set_input_to_output(state)
        self.model = keras.Model(inputs = inputs, outputs = outputs)
        self.model.summary()

    def set_optimizer_loss(self) :
        if self.config.grad_clip :
            self.optimizer = keras.optimizers.Adam(clipnorm= 10)
        else :
            self.optimizer = keras.optimizers.Adam()
        self.mse = keras.losses.MeanSquaredError()

    def initialize(self):
        self.set_summary()
        
    def save(self) :
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)
        self.model.save(os.path.join(self.config.model_output, 'Model.h5'))

    def set_summary(self) :
        self.file_writer = tf.summary.create_file_writer(self.config.output_path)
        self.file_writer.set_as_default()
        self.callback = keras.callbacks.TensorBoard(log_dir= self.config.output_path)

    def add_summary(self, t) :
        tf.summary.scalar("loss", self.loss, step=t)
        tf.summary.scalar("grads_norm", self.grad_norm,step=t)

        tf.summary.scalar("Avg_Reward", self.avg_reward,step=t)
        tf.summary.scalar("Max_Reward", self.max_reward,step=t)
        tf.summary.scalar("Std_Reward", self.std_reward,step=t)

        tf.summary.scalar("Avg_Q", self.avg_q,step=t)
        tf.summary.scalar("Max_Q", self.max_q,step=t)
        tf.summary.scalar("Std_Q", self.std_q,step=t)

        tf.summary.scalar("Eval_Reward", self.eval_reward,step=t)
        
    def loss_func(self, q, target_q) :
        """
        return loss
        """
        num_actions = self.env.action_space.n
        Q_samp = self.r + tf.cast(tf.math.logical_not(self.done_mask), tf.float32)* self.config.gamma*tf.math.reduce_max(target_q,axis=1)
        mask = tf.one_hot(self.a, num_actions, dtype= q.dtype)
        Q_sa = tf.math.reduce_sum(q * mask, axis = 1)
        return self.mse(Q_samp, Q_sa)
    
    def update_step(self, t, replay_buffer, lr) :

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)

        self.s = s_batch
        self.a = a_batch
        self.r = r_batch
        self.sp = sp_batch
        self.done_mask = done_mask_batch
        self.lr = lr

        with tf.GradientTape() as tape :
            q = self.model(self.s)
            self.loss = self.loss_func(q, self.target_model(self.sp))
        gradients = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.learning_rate = self.lr
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.grad_norm = tf.linalg.global_norm(gradients)

        self.add_summary(t)

        return self.loss, self.grad_norm

    def update_target_params(self) :
        self.update_target()