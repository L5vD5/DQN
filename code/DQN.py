import datetime,gym,os,pybullet_envs,time,psutil,ray
import random

import numpy as np
import tensorflow as tf
from config import Config
from collections import deque

print(tf.executing_eagerly())

print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

class DQNAgent():
    """
    Deep Q Network Model
    """
    def __init__(self, odim, adim, name="main"):
        # self.config = Config()
        # self.env, self.eval_env = get_envs()
        # self.odim = self.env.observation_space.shape[0]
        # self.adim = self.env.action_space.shape[0]
        self.odim = odim
        self.adim = adim
        # self.buf = deque()

        # self.model = tf.keras.Model(tf.keras.Input(shape=(1, self.odim)), tf.keras.Input(shape=(1, self.adim)), self.create_dqn_model)

    def mlp(self, hdims=[64,64], actv='relu', output_actv=None):
        model = tf.keras.Sequential()
        model.add(self.input(self.odim))
        for hdim in hdims[:-1]:
            model.add(tf.keras.layers.Dense(hdim, activation=actv))
        model.add(tf.keras.layers.Dense(self.adim, activation=output_actv, trainable=False))
        return model

    def input(self, dim=None):
        return tf.keras.Input(dtype=tf.float32, shape=(None, dim) if dim else (None,))

    def predict(self, o):
        x = np.reshape(o, [1, self.odim])
        self.model = self.mlp(config.hdims)
        self._Qpred = self.model(x)
        return self._Qpred

    def update(self, x_stack, y_stack):
        self._loss = lambda: tf.keras.losses.mse(y_stack, self.Qpred)
        opt = tf.keras.optimizers.Adam(learning_rate=self.config.lr)
        self._train = opt.minimize(loss=self._loss, var_list=self.model.trainable_variables)
        return self._train

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

def main():
    start_time = time.time()
    config = Config()
    env, eval_env = get_envs()
    odim = env.observation_space.shape[0]
    adim = env.action_space.shape[0]

    mainDQN = DQNAgent(odim, adim, name='main')
    targetDQN = DQNAgent(odim, adim, name='target')

    o, r, d, ep_ret, ep_len, n_env_step = eval_env.reset(), 0, False, 0, 0, 0
    for epoch in range(config.epochs):
        o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
        while not(d or (ep_len == config.steps_per_epoch)):
            a = eval_env.action_space.sample()
            Q = mainDQN.predict(o)
            o1, r, d, _ = eval_env.step(Q.numpy()[0])
            ep_len += 1
            ep_ret += r
            n_env_step += 1

            # Save the Experience to our buffer
            replay_buffer.append((o, a, r, o1,d))
            if len(replay_buffer) > 50000:
                replay_buffer.popleft()
            o = o1
        # Evaluate
        if (epoch == 0) or (((epoch + 1) % config.evaluate_every) == 0):
            ram_percent = psutil.virtual_memory().percent  # memory usage
            print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                  (epoch + 1, config.epochs, epoch / config.epochs * 100,
                   n_env_step,
                   time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                   ram_percent)
                  )
            for _ in range(50):
                minibatch = random.sample(replay_buffer, 10)
                loss, _ = simple_replay_train(mainDQN, minibatch)
            o, d, ep_ret, ep_len = eval_env.reset(), False, 0, 0
            _ = eval_env.render(mode='human')
            while not (d or (ep_len == config.steps_per_epoch)):
                # a = sess.run(model['mu'], feed_dict={model['o_ph']: o.reshape(1, -1)})
                o, r, d, _ = eval_env.step(a)
                _ = eval_env.render(mode='human')
                ep_ret += r  # compute return
                ep_len += 1
            print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))

    # def observe():



main()
