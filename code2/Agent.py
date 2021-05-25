import datetime,gym,os,pybullet_envs,time,psutil,ray
from ReplayBuffer import ReplayBuffer
from DQN import DQNNetwork
import tensorflow as tf
import numpy as np
from config import Config

class Agent(object):
    def __init__(self, ):
        # Config
        self.config = Config()

        # Environment
        self.env, self.eval_env = get_envs()
        self.odim = self.env.observation_space.shape[0]
        self.adim = self.env.action_space.shape[0]

        # Network
        self.main_network = DQNNetwork(self.odim, self.adim)
        self.target_network = DQNNetwork(self.odim, self.adim)
        self.gamma = self.config.gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-6)
        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")
        self.eps = 0.3

        # Buffer (Memory)
        self.buffer = ReplayBuffer(buffer_size=self.config.buffer_size, odim=self.odim, adim=self.adim, batch_size=self.config.mini_batch_size)


    @tf.function
    def getQ(self, obs):
        Q=self.main_network(obs)
        return Q

    # @tf.function
    def update_main_network(self, o_batch, a_batch, r_batch, o1_batch, d_batch):
        # print('update main network', o_batch, a_batch, r_batch, o1_batch, d_batch)

        with tf.GradientTape() as tape:
            o1_q = self.target_network(tf.constant(value=o1_batch))
            max_o1_q = tf.reduce_max(o1_q, axis=1)
            expected_q = r_batch + self.gamma * max_o1_q * (1-tf.constant(d_batch.astype('float32')))
            main_q = tf.reduce_sum(self.main_network(o_batch) )
            loss = tf.keras.losses.mse(tf.stop_gradient(expected_q), main_q)

        gradients = tape.gradient(loss, self.main_network.trainable_weights)
        # clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.main_network.trainable_variables))

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        return loss

    # @tf.function
    def update_target_network(self):
        print('update target network', self.target_network)
        self.target_network.set_weights(self.main_network.get_weights())


    def train(self):
        start_time = time.time()

        o, r, d, ep_ret, ep_len, n_env_step = self.env.reset(), 0, False, 0, 0, 0
        for epoch in range(self.config.epochs):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not (d or (ep_len == self.config.steps_per_epoch)):
                # Q = self.getQ(o.reshape(1, -1))
                Q = self.getQ(tf.constant(value=o.reshape(1, -1)))
                a = Q.numpy()[0]
                o1, r, d, _ = self.env.step(a)
                ep_len += 1
                ep_ret += r
                n_env_step += 1

                # Save the Experience to our buffer
                self.buffer.append(o, a, r, o1, d)
                o = o1

            # Update main network
            o_batch, a_batch, r_batch, o1_batch, d_batch = self.buffer.sample()
            self.update_main_network(o_batch, a_batch, r_batch, o1_batch, d_batch)

            # Evaluate
            if (epoch == 0) or (((epoch + 1) % self.config.evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, self.config.epochs, epoch / self.config.epochs * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                # Update target network
                self.update_target_network()
                o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                _ = self.eval_env.render(mode='human')
                while not (d or (ep_len == self.config.steps_per_epoch)):
                    # a = sess.run(model['mu'], feed_dict={model['o_ph']: o.reshape(1, -1)})
                    Q = self.getQ(tf.constant(value=o.reshape(1, -1)))
                    a = Q.numpy()[0]
                    o, r, d, _ = self.eval_env.step(a)
                    _ = self.eval_env.render(mode='human')
                    ep_ret += r  # compute return
                    ep_len += 1
                print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))

    # @tf.function
    # def learn(self):
    #     q_target = rewards + (1- dones) * self.gamma * tf.reduce_max(self.target(next_o))

def get_envs():
    env_name = 'AntBulletEnv-v0'
    env,eval_env = gym.make(env_name),gym.make(env_name)
    _ = eval_env.render(mode='human') # enable rendering on test_env
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return env,eval_env

a = Agent()
a.train()