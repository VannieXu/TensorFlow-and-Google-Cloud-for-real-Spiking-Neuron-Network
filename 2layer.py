import tensorflow as tf
import numpy as np
import itertools
import scipy.spatial.distance
from scipy import signal

x = list(itertools.product(range(48),repeat=2))
d = scipy.spatial.distance.pdist(x,'sqeuclidean')
y = scipy.spatial.distance.squareform(d)
y2 = y[::2]
z = np.reshape(range(1152),(48,24))
z = np.reshape(z[::2],(576))
y2 = y2[z]
Pee_granular = 0.15*np.exp(-(y/4)**2)
Pie_granular = 0.3*np.exp(-(y2/4)**2)
Pee_supra = 0.15*np.exp(-(y/4)**2)
Pie_supra = 0.3*np.exp(-(y2/4)**2)
rand_granular_ee = np.random.rand(2304,2304)
rand_granular_ie = np.random.rand(576,2304)
rand_supra_ee = np.random.rand(2304,2304)
rand_supra_ie = np.random.rand(576,2304)
x1,y1 = np.where(Pee_granular-rand_granular_ee<0)
Pee_granular[x1,y1] = 0
Pee_granular = Pee_granular*5
np.fill_diagonal(Pee_granular,0)
x1,y1 = np.where(Pie_granular-rand_granular_ie<0)
Pie_granular[x1,y1] = 0
x1,y1 = np.where(Pee_supra-rand_supra_ee<0)
Pee_supra[x1,y1] = 0
Pee_supra = Pee_supra*5
np.fill_diagonal(Pee_supra,0)
x1,y1 = np.where(Pie_supra-rand_supra_ie<0)
Pie_supra[x1,y1] = 0

x = list(itertools.product(range(24),repeat=2))
d = scipy.spatial.distance.pdist(x,'sqeuclidean')
y = scipy.spatial.distance.squareform(d)
Pii_granular = 0.5*np.exp(-(y/2)**2)
Pii_supra = 0.5*np.exp(-(y/2)**2)
rand_granular_ii = np.random.rand(576,576)
rand_supra_ii = np.random.rand(576,576)
x1,y1 = np.where(Pii_granular-rand_granular_ii<0)
Pii_granular[x1,y1]=0
np.fill_diagonal(Pii_granular,0)
x1,y1 = np.where(Pii_supra-rand_supra_ii<0)
Pii_supra[x1,y1]=0
np.fill_diagonal(Pii_supra,0)

x = list((0,i) for i in range(48))
y = list((1,i) for i in range(48))
d = scipy.spatial.distance_matrix(x,y)
Pee_vertical = 0.8*np.exp(-(d/2)**2)
rand_vertical_ee = np.random.rand(48,48)
x1,y1 = np.where(Pee_vertical-rand_vertical_ee<0)
Pee_vertical[x1,y1]=0
y = y[::2]
d = scipy.spatial.distance_matrix(x,y)
Pie_vertical = 0.5*np.exp(-(d/2)**2)
rand_vertical_ie = np.random.rand(48,24)
x1,y1 = np.where(Pie_vertical-rand_vertical_ie<0)
Pie_vertical[x1,y1]=0

sess = tf.InteractiveSession()

class SynapticDelay(tf.contrib.rnn.RNNCell):

    def __init__(self,num_units):
        self._state_size = num_units

    def __call__(self, inputs, state, scope=None):
        output = state[:,-1]
        newstate = tf.concat([inputs,state[:,0:-1]],1)
        return output,newstate

class NewCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_state):
        self._state_size = num_state

    def __call__(self, inputs, state, scope=None):
        Iext = inputs
        V, G, F, W, B = tf.split(state, [1, 1, 1, 1, 1],1)
        dt = 0.01
        g_Na = 120
        g_K = 15
        g_L = 0.3
        g_A = 9
        V_Na = 55
        V_K = -72
        V_L = -53
        V_syn = 55
        taus = 2
        synth = 10

        def Steady_Variable(a, V, V_half):
            P = 1 / (1 + tf.exp(-2 * a * (V - V_half)))
            return P

        def TimeConstantW(Lam, V, V_half):
            tao = 1 / (Lam * (tf.exp(0.055 * (V - V_half)) + tf.exp(-0.055 * (V - V_half))))
            return tao

        Im = Steady_Variable(0.065, V, -31)
        IW = Steady_Variable(0.055, V, -35)
        IA = Steady_Variable(0.020, V, -20)
        IB = Steady_Variable(-0.10, V, -70)
        taow = TimeConstantW(0.08, V, -35)

        W_ = W + dt * (IW - W) / taow
        B_ = B + dt * (IB - B) / 10
        F_ = F + dt / taus * (-F + (tf.sign(V - synth) + 1) / 2)
        G_ = G + dt / taus * (-G + F)

        I_Na = g_Na * Im ** 3 * (1 - W) * (V - V_Na)
        I_K = g_K * W ** 4 * (V - V_K)
        I_A = g_A * IA * B * (V - V_K)
        I_L = g_L * (V - V_L)
        SumI = Iext - I_Na - I_K - I_A - I_L
        V_ = V + dt * SumI
        output = G*(V-V_syn)

        nextstate = tf.concat([V_,G_,F_,W_,B_],1)

        return output,nextstate

cell = NewCell(5)
delay = SynapticDelay(100)

granular_pyramidal_input = tf.placeholder(shape=(2304,1),dtype=tf.float32)
granular_pyramidal_state = tf.placeholder(shape=(2304,5),dtype=tf.float32)
granular_pyramidal_delay = tf.placeholder(shape=(2304,300),dtype=tf.float32)

granular_basket_input = tf.placeholder(shape=(576,1),dtype=tf.float32)
granular_basket_state = tf.placeholder(shape=(576,5),dtype=tf.float32)
granular_basket_delay = tf.placeholder(shape=(576,1200),dtype=tf.float32)

supra_pyramidal_input = tf.placeholder(shape=(2304,1),dtype=tf.float32)
supra_pyramidal_state = tf.placeholder(shape=(2304,5),dtype=tf.float32)
supra_pyramidal_delay = tf.placeholder(shape=(2304,300),dtype=tf.float32)

supra_basket_input = tf.placeholder(shape=(576,1),dtype=tf.float32)
supra_basket_state = tf.placeholder(shape=(576,5),dtype=tf.float32)
supra_basket_delay = tf.placeholder(shape=(576,1200),dtype=tf.float32)

(granular_pyramidal_output,granular_pyramidal_nextstate) = cell(granular_pyramidal_input,granular_pyramidal_state)
(granular_pyramidal_synout,granular_pyramidal_synstate) = delay(granular_pyramidal_output,granular_pyramidal_delay)

(granular_basket_output,granular_basket_nextstate) = cell(granular_basket_input,granular_basket_state)
(granular_basket_synout,granular_basket_synstate) = delay(granular_basket_output,granular_basket_delay)

(supra_pyramidal_output,supra_pyramidal_nextstate) = cell(supra_pyramidal_input,supra_pyramidal_state)
(supra_pyramidal_synout,supra_pyramidal_synstate) = delay(supra_pyramidal_output,supra_pyramidal_delay)

(supra_basket_output,supra_basket_nextstate) = cell(supra_basket_input,supra_basket_state)
(supra_basket_synout,supra_basket_synstate) = delay(supra_basket_output,supra_basket_delay)

granular_ee = tf.layers.dense(tf.reshape(granular_pyramidal_synout,(1,2304)),2304,kernel_initializer=tf.constant_initializer(Pee_granular))
granular_ei = tf.layers.dense(tf.reshape(granular_pyramidal_synout,(1,2304)),576,kernel_initializer=tf.constant_initializer(Pie_granular))
granular_ie = tf.layers.dense(tf.reshape(granular_basket_synout,(1,576)),2304,
                              kernel_initializer=tf.constant_initializer(-np.transpose(Pie_granular)))
granular_ii = tf.layers.dense(tf.reshape(granular_basket_synout,(1,576)),576,kernel_initializer=tf.constant_initializer(-Pii_granular))

supra_ee = tf.layers.dense(tf.reshape(supra_pyramidal_synout,(1,2304)),2304,kernel_initializer=tf.constant_initializer(Pee_supra))
supra_ei = tf.layers.dense(tf.reshape(supra_pyramidal_synout,(1,2304)),576,kernel_initializer=tf.constant_initializer(Pie_supra))
supra_ie = tf.layers.dense(tf.reshape(supra_basket_synout,(1,576)),2304,
                              kernel_initializer=tf.constant_initializer(-np.transpose(Pie_supra)))
supra_ii = tf.layers.dense(tf.reshape(supra_basket_synout,(1,576)),576,kernel_initializer=tf.constant_initializer(-Pii_supra))

granular_supra_ee = tf.layers.dense(tf.reshape(granular_pyramidal_synout,(48,48)),48,kernel_initializer=tf.constant_initializer(Pee_vertical))
Layer_ei = tf.reshape(granular_pyramidal_synout,(48,48))
granular_supra_ei = tf.layers.dense(tf.transpose(Layer_ei[:,::2]),24,
                                    kernel_initializer=tf.constant_initializer(np.transpose(Pie_vertical)))

granular_pyramidal = tf.add(granular_ee,granular_ie)
granular_basket = tf.add(granular_ei,granular_ii)

supra_pyramidal = tf.add(tf.add(supra_ee,supra_ie),tf.reshape(granular_supra_ee,(1,2304)))
supra_basket = tf.add(tf.add(supra_ii,supra_ei),tf.reshape(granular_supra_ei,(1,576)))

Input_granular_pyramidal = np.zeros((2304,1))
y = [48*i+23 for i in range(48)]
Input_granular_pyramidal[y] = 8
Input_granular_basket = np.zeros((576,1))
y = [24*i+11 for i in range(24)]
Input_granular_basket[y] = 8
Input_supra_pyramidal = np.zeros((2304,1))
Input_supra_basket = np.zeros((576,1))

V_init = -60
G_init = 0
F_init = 0
W_init = 1 / (1 + np.exp(-2 * 0.055 * (V_init + 35)))
B_init = 1 / (1 + np.exp(2 * 0.1 * (V_init + 70)))
cell_init = np.array([V_init,G_init,F_init,W_init,B_init])

state_granular_pyramidal = np.matrix([cell_init for _ in range(2304)])
state_granular_basket = np.matrix([cell_init for _ in range(576)])
state_supra_pyramidal = np.matrix([cell_init for _ in range(2304)])
state_supra_basket = np.matrix([cell_init for _ in range(576)])

delay_granular_pyramidal = np.zeros((2304,300))
delay_granular_basket = np.zeros((576,1200))
delay_supra_pyramidal = np.zeros((2304,300))
delay_supra_basket = np.zeros((576,1200))

sess.run(tf.global_variables_initializer())

P_granular_pyramidal = []
P_granular_basket = []
P_supra_pyramidal = []
P_supra_basket = []
state_granular_pyramidal,delay_granular_pyramidal,state_granular_basket,delay_granular_basket,state_supra_pyramidal,delay_supra_pyramidal,\
state_supra_basket,delay_supra_basket,Input_granular_pyramidal2,Input_granular_basket2,Input_supra_pyramidal2,Input_supra_basket2 = \
sess.run([granular_pyramidal_nextstate,granular_pyramidal_synstate,granular_basket_nextstate,granular_basket_synstate,
          supra_pyramidal_nextstate,supra_pyramidal_synstate,supra_basket_nextstate,supra_basket_synstate,granular_pyramidal,granular_basket,
          supra_pyramidal,supra_basket],
         feed_dict={granular_pyramidal_input:Input_granular_pyramidal,granular_pyramidal_state:state_granular_pyramidal,
                    granular_pyramidal_delay:delay_granular_pyramidal,granular_basket_input:Input_granular_basket,
                    granular_basket_state:state_granular_basket,granular_basket_delay:delay_granular_basket,
                    supra_pyramidal_input:Input_supra_pyramidal,supra_pyramidal_state:state_supra_pyramidal,supra_pyramidal_delay:delay_supra_pyramidal,
                    supra_basket_input:Input_supra_basket,supra_basket_state:state_supra_basket,supra_basket_delay:delay_supra_basket})
P_granular_pyramidal.append(state_granular_pyramidal[:,0])
P_granular_basket.append(state_granular_basket[:,0])
P_supra_pyramidal.append(state_supra_pyramidal[:,0])
P_supra_basket.append(state_supra_basket[:,0])

for _ in range(29999):
    Input_granular_pyramidal1 = Input_granular_pyramidal - np.transpose(Input_granular_pyramidal2)
    Input_granular_basket1 = Input_granular_basket - np.transpose(Input_granular_basket2)
    Input_supra_pyramidal = -np.transpose(Input_supra_pyramidal2)
    Input_supra_basket = -np.transpose(Input_supra_basket2)
    state_granular_pyramidal, delay_granular_pyramidal, state_granular_basket, delay_granular_basket, state_supra_pyramidal, delay_supra_pyramidal, \
    state_supra_basket, delay_supra_basket, Input_granular_pyramidal2, Input_granular_basket2, Input_supra_pyramidal2, Input_supra_basket2 = \
        sess.run([granular_pyramidal_nextstate, granular_pyramidal_synstate, granular_basket_nextstate,
                  granular_basket_synstate,
                  supra_pyramidal_nextstate, supra_pyramidal_synstate, supra_basket_nextstate, supra_basket_synstate,
                  granular_pyramidal, granular_basket,
                  supra_pyramidal, supra_basket],
                 feed_dict={granular_pyramidal_input: Input_granular_pyramidal1,
                            granular_pyramidal_state: state_granular_pyramidal,
                            granular_pyramidal_delay: delay_granular_pyramidal,
                            granular_basket_input: Input_granular_basket1,
                            granular_basket_state: state_granular_basket, granular_basket_delay: delay_granular_basket,
                            supra_pyramidal_input: Input_supra_pyramidal, supra_pyramidal_state: state_supra_pyramidal,
                            supra_pyramidal_delay: delay_supra_pyramidal,
                            supra_basket_input: Input_supra_basket, supra_basket_state: state_supra_basket,
                            supra_basket_delay: delay_supra_basket})
    P_granular_pyramidal.append(state_granular_pyramidal[:, 0])
    P_granular_basket.append(state_granular_basket[:, 0])
    P_supra_pyramidal.append(state_supra_pyramidal[:, 0])
    P_supra_basket.append(state_supra_basket[:, 0])

P_granular_pyramidal = np.array(P_granular_pyramidal)
P_granular_basket = np.array(P_granular_basket)
P_supra_pyramidal = np.array(P_supra_pyramidal)
P_supra_basket = np.array(P_supra_basket)

Spike_granular_pyramidal = np.zeros((2304,30))
Spike_granular_basket = np.zeros((576,30))
Spike_supra_pyramidal = np.zeros((2304,30))
Spike_supra_basket = np.zeros((576,30))

for i in range(30):
    for j in range(2304):
        ind,_=signal.find_peaks(P_granular_pyramidal[i*1000:(i+1)*1000,j],height=50)
        Spike_granular_pyramidal[j,i]=Spike_granular_pyramidal[j,i]+ind.size
        ind,_=signal.find_peaks(P_supra_pyramidal[i*1000:(i+1)*1000,j],height=50)
        Spike_supra_pyramidal[j,i]=Spike_supra_pyramidal[j,i]+ind.size

for i in range(30):
    for j in range(576):
        ind,_=signal.find_peaks(P_granular_basket[i*1000:(i+1)*1000,j],height=50)
        Spike_granular_basket[j,i]=Spike_granular_basket[j,i]+ind.size
        ind,_=signal.find_peaks(P_supra_basket[i*1000:(i+1)*1000,j],height=50)
        Spike_supra_basket[j,i]=Spike_supra_basket[j,i]+ind.size


np.savetxt('GP.csv',Spike_granular_pyramidal)
np.savetxt('GB.csv',Spike_granular_basket)
np.savetxt('SP.csv',Spike_supra_pyramidal)
np.savetxt('SB.csv',Spike_supra_basket)
