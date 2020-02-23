import tensorflow as tf
from tensorflow import keras

from utils.general import get_logger
from utils.test_env import EnvTest
from q1_schedule import LinearExploration, LinearSchedule
from q2_linear import Linear

from configs.q3_nature import config

class NatureQN(Linear) :
    def set_input_to_output(self, inputs) :
        num_actions = self.env.action_space.n
        x = keras.layers.Conv2D(filters = 32, kernel_size = (8,8),strides = 4, activation= 'relu')(inputs)
        x = keras.layers.Conv2D(filters = 64, kernel_size = (4,4),strides = 2, activation= 'relu')(x)
        x = keras.layers.Conv2D(filters = 64, kernel_size = (3,3),strides = 1, activation= 'relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units = 512, activation = 'relu')(x)
        return keras.layers.Dense(units = num_actions)(x)

if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)
