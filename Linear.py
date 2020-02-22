import tensorflow as tf
from tensorflow import keras
from utils.general import get_logger
from utils.test_env import EnvTest
from DQL import DQN
from q1_schedule import LinearExploration, LinearSchedule
from configs.q2_linear import config

class Linear(DQN):
    def set_input_to_output(self, inputs) :
        num_actions = self.env.action_space.n
        x = keras.layers.Flatten()(inputs)
        return keras.layers.Dense(num_actions)(x)

if __name__ == '__main__':
    env = EnvTest((5, 5, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = Linear(env, config)
    model.run(exp_schedule, lr_schedule)
