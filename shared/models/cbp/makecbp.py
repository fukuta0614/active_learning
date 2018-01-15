import numpy as np

d = 512
c = 512
W1 = np.random.choice([-1, 1], (d, c)).astype(np.float32)[:, :, np.newaxis, np.newaxis]
W2 = np.random.choice([-1, 1], (d, c)).astype(np.float32)[:, :, np.newaxis, np.newaxis]
np.savez('randweight_{}_to_{}.npz'.format(c, d), W1=W1, W2=W2)
