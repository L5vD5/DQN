from DQN import DQN

class Agent(object):
    def __init__(self, env, steps, gamma=0.99, epsilon=1.0, epsilon_dacay=0.999,
                 buffer_size=2000, batch_size=64, target_update_step=100):
        self.env = env
        # self.args = args
        self.odim = odim
        self.adim = adim
        self.steps = steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dacay = epsilon_dacay
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.target_update_step = target_update_step

        # self.replay_buffer
        self.main = DQN(name='main')
        self.target = DQN(name='target')
        print("Test")
        network = self.main.policy()


    # @tf.function
    # def learn(self):
    #     q_target = rewards + (1- dones) * self.gamma * tf.reduce_max(self.target(next_o))
