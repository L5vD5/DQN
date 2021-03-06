# Configuration, set model parameter
class Config:
    def __init__(self):
        # Model
        self.hdims = [256, 256]
        #Graph
        self.clip_ratio = 0.2
        self.lr = 3e-4
        #Buffer
        self.steps_per_epoch = 5000
        self.gamma = 0.99
        self.lam = 0.95
        self.buffer_size = 10000
        self.mini_batch_size = 32
        self.eps = 1.0
        #Update
        self.train_pi_iters = 100
        self.train_v_iters = 100
        self.target_kl = 0.01
        self.epochs = 10000
        self.max_ep_len = 10000
        self.print_every = 10
        self.evaluate_every = 10
        self.update_every = 10
