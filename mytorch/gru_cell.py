import numpy as np
from activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t
        
        # Compute reset gate
        self.r = self.r_act(np.dot(x, self.Wrx.T) + self.brx + np.dot(self.hidden, self.Wrh.T) + self.brh)
        
        # Compute update gate
        self.z = self.z_act(np.dot(x, self.Wzx.T) + self.bzx + np.dot(self.hidden, self.Wzh.T) + self.bzh)
        
        # Compute new gate
        self.n = self.h_act(np.dot(x, self.Wnx.T) + self.bnx + self.r * (np.dot(self.hidden, self.Wnh.T) + self.bnh))
        #self.ztemp=self.r*h_prev_t
        # Compute hidden state
        h_t = (1 - self.z) * self.n + self.z * self.hidden
        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        # print('selfxshape', self.x.shape)
        # print('selfdshape', self.d)
        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,) # h_t is the final output of you GRU cell.
        
        # return h_t
        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        
        # Compute the derivatives of the output gate.

        self.x = np.reshape(self.x, (self.d, 1))
        self.hidden = np.reshape(self.hidden, (self.h, 1))
        delta = np.reshape(delta, (self.h, 1))
        self.r = np.reshape(self.r, (self.h, 1))
        self.n = np.reshape(self.n, (self.h, 1))
        self.z = np.reshape(self.z, (self.h, 1))

        dn = delta * (1-self.z)
        dz = delta * (-self.n + self.hidden)
        dr = dn * self.h_act.backward(state=self.n) * (np.dot(self.Wnh, self.hidden) + self.bnh.reshape(self.h, 1))

        dr = dr*np.expand_dims(self.r_act.backward(), 1)
        self.dWrx = np.dot(dr, self.x.T)
        self.dWrh = np.dot(dr, self.hidden.T)
        self.dbrx = np.reshape(dr, (self.h,))
        self.dbrh = np.reshape(dr, (self.h,))

        dz = dz * np.expand_dims(self.z_act.backward(), 1)
        self.dWzx = np.dot(dz, self.x.T)
        self.dWzh = np.dot(dz, self.hidden.T)
        self.dbzx = np.reshape(dz, (self.h,))
        self.dbzh = np.reshape(dz, (self.h,))

        dn = dn * self.h_act.backward(self.n)
        self.dWnx = np.dot(dn, self.x.T)
        self.dWnh = np.dot(dn*self.r, self.hidden.T)
        self.dbnx = np.reshape(dn, (self.h,))
        self.dbnh = np.reshape(dn*self.r, (self.h,))

        dx = np.dot(dn.T, self.Wnx) + np.dot(dz.T, self.Wzx) + np.dot(dr.T, self.Wrx)
        dh = (delta*self.z).T + np.dot((dn*self.r).T, self.Wnh) + np.dot(dz.T, self.Wzh) + np.dot(dr.T, self.Wrh)

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        #return dx_t, dh
