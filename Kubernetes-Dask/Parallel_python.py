import numpy as np
import itertools
import scipy.spatial.distance

import dask
import time

from dask.distributed import Client, config

config['scheduler-address']
client = Client()
info = client.scheduler_info()
workers = iter(info['workers'])
addr = []
for i in range(len(info['workers'])):
    w = next(workers)
    addr.append(w)
    
def weight(inter, exc):
    n_inter = inter * inter
    n_exc = exc * exc

    x = list(itertools.product(range(exc), repeat=2))
    d = scipy.spatial.distance.pdist(x, 'sqeuclidean')
    y = scipy.spatial.distance.squareform(d)
    y2 = y[::2]
    z = np.reshape(range(inter * exc), (exc, inter))
    z = np.reshape(z[::2], (n_inter))
    y2 = y2[z]
    Pee_granular = 0.15 * np.exp(-(y / 4) ** 2)
    Pie_granular = 0.3 * np.exp(-(y2 / 4) ** 2)
    Pee_supra = 0.15 * np.exp(-(y / 4) ** 2)
    Pie_supra = 0.3 * np.exp(-(y2 / 4) ** 2)
    rand_granular_ee = np.random.rand(n_exc, n_exc)
    rand_granular_ie = np.random.rand(n_inter, n_exc)
    rand_supra_ee = np.random.rand(n_exc, n_exc)
    rand_supra_ie = np.random.rand(n_inter, n_exc)
    x1, y1 = np.where(Pee_granular - rand_granular_ee < 0)
    Pee_granular[x1, y1] = 0
    Pee_granular = Pee_granular * 5
    np.fill_diagonal(Pee_granular, 0)
    x1, y1 = np.where(Pie_granular - rand_granular_ie < 0)
    Pie_granular[x1, y1] = 0
    x1, y1 = np.where(Pee_supra - rand_supra_ee < 0)
    Pee_supra[x1, y1] = 0
    Pee_supra = Pee_supra * 5
    np.fill_diagonal(Pee_supra, 0)
    x1, y1 = np.where(Pie_supra - rand_supra_ie < 0)
    Pie_supra[x1, y1] = 0

    x = list(itertools.product(range(inter), repeat=2))
    d = scipy.spatial.distance.pdist(x, 'sqeuclidean')
    y = scipy.spatial.distance.squareform(d)
    Pii_granular = 0.5 * np.exp(-(y / 2) ** 2)
    Pii_supra = 0.5 * np.exp(-(y / 2) ** 2)
    rand_granular_ii = np.random.rand(n_inter, n_inter)
    rand_supra_ii = np.random.rand(n_inter, n_inter)
    x1, y1 = np.where(Pii_granular - rand_granular_ii < 0)
    Pii_granular[x1, y1] = 0
    np.fill_diagonal(Pii_granular, 0)
    x1, y1 = np.where(Pii_supra - rand_supra_ii < 0)
    Pii_supra[x1, y1] = 0
    np.fill_diagonal(Pii_supra, 0)

    x = list((0, i) for i in range(exc))
    y = list((1, i) for i in range(exc))
    d = scipy.spatial.distance_matrix(x, y)
    Pee_vertical = 0.8 * np.exp(-(d / 2) ** 2)
    rand_vertical_ee = np.random.rand(exc, exc)
    x1, y1 = np.where(Pee_vertical - rand_vertical_ee < 0)
    Pee_vertical[x1, y1] = 0
    y = y[::2]
    d = scipy.spatial.distance_matrix(x, y)
    Pie_vertical = 0.5 * np.exp(-(d / 2) ** 2)
    rand_vertical_ie = np.random.rand(exc, inter)
    x1, y1 = np.where(Pie_vertical - rand_vertical_ie < 0)
    Pie_vertical[x1, y1] = 0

    w_la = Pee_granular, Pie_granular, Pii_granular, Pee_supra, Pie_supra, Pii_supra
    w_ver = Pee_vertical, Pie_vertical
    return w_la, w_ver

def cell_dynamic(initial_value,Iext):
    V, G, F, W, B = initial_value[:,0],initial_value[:,1],initial_value[:,2],initial_value[:,3],initial_value[:,4]
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
        P = 1 / (1 + np.exp(-2 * a * (V - V_half)))
        return P

    def TimeConstantW(Lam, V, V_half):
        tao = 1 / (Lam * (np.exp(0.055 * (V - V_half)) + np.exp(-0.055 * (V - V_half))))
        return tao

    Im = Steady_Variable(0.065, V, -31)
    IW = Steady_Variable(0.055, V, -35)
    IA = Steady_Variable(0.020, V, -20)
    IB = Steady_Variable(-0.10, V, -70)
    taow = TimeConstantW(0.08, V, -35)

    W = W + dt * (IW - W) / taow
    B = B + dt * (IB - B) / 10
    F = F + dt / taus * (-F + (np.sign(V - synth) + 1) / 2)
    G = G + dt / taus * (-G + F)

    output = G * (V - V_syn)
    I_Na = g_Na * Im ** 3 * (1 - W) * (V - V_Na)
    I_K = g_K * W ** 4 * (V - V_K)
    I_A = g_A * IA * B * (V - V_K)
    I_L = g_L * (V - V_L)
    SumI = Iext - I_Na - I_K - I_A - I_L
    V = V + dt * SumI

    return np.transpose(np.array([V,G,F,W,B])),output

def Synaptic_Delay(Current,Syn_in):
    output = Current[:,-1]
    nextstate = np.concatenate((np.transpose(np.array([Syn_in])),Current[:,0:-1]),axis=1)
    return output,nextstate
  
def main(i,e):
    inter = i
    exc = e

    n_inter = inter*inter
    n_exc = exc*exc

    V_init = -60
    G_init = 0
    F_init = 0
    W_init = 1 / (1 + np.exp(-2 * 0.055 * (V_init + 35)))
    B_init = 1 / (1 + np.exp(2 * 0.1 * (V_init + 70)))
    cell_init = np.array([V_init, G_init, F_init, W_init, B_init])

    P_granular_pyramidal = []
    P_granular_basket = []
    P_supra_pyramidal = []
    P_supra_basket = []

    state_granular_pyramidal = np.array([cell_init for _ in range(n_exc)])
    state_granular_basket = np.array([cell_init for _ in range(n_inter)])
    state_supra_pyramidal = np.array([cell_init for _ in range(n_exc)])
    state_supra_basket = np.array([cell_init for _ in range(n_inter)])

    delay_granular_pyramidal = np.zeros((n_exc,300))
    delay_granular_basket = np.zeros((n_inter,1200))
    delay_supra_pyramidal = np.zeros((n_exc,300))
    delay_supra_basket = np.zeros((n_inter,1200))

    Input_granular_pyramidal = np.zeros(n_exc)
    y = [int(exc*i+(exc/2-1)) for i in range(exc)]
    Input_granular_pyramidal[y] = 8
    Input_granular_basket = np.zeros(n_inter)
    y = [int(inter*i+(inter/2-1)) for i in range(inter)]
    Input_granular_basket[y] = 8
    Input_supra_pyramidal = np.zeros(n_exc)
    Input_supra_basket = np.zeros(n_inter)

    Input_granular_pyramidal1 = Input_granular_pyramidal
    Input_granular_basket1 = Input_granular_basket

    Pee_granular, Pie_granular, Pii_granular, Pee_supra, Pie_supra, Pii_supra = weight(inter,exc)[0]
    Pee_vertical, Pie_vertical = weight(inter,exc)[1]
    
    for i in range(5000):
        state_granular_pyramidal, granular_pyramidal_output = cell_dynamic(state_granular_pyramidal,Input_granular_pyramidal1)
        state_granular_basket, granular_basket_output = cell_dynamic(state_granular_basket,Input_granular_basket1)
        state_supra_pyramidal, supra_pyramida_output = cell_dynamic(state_supra_pyramidal,Input_supra_pyramidal)
        state_supra_basket, supra_basket_output = cell_dynamic(state_supra_basket,Input_supra_basket)

        granular_pyramidal_synout, delay_granular_pyramidal = Synaptic_Delay(delay_granular_pyramidal,granular_pyramidal_output)
        granular_basket_synout, delay_granular_basket = Synaptic_Delay(delay_granular_basket,granular_basket_output)
        supra_pyramidal_synout, delay_supra_pyramidal = Synaptic_Delay(delay_supra_pyramidal,supra_pyramida_output)
        supra_basket_synout, delay_supra_basket = Synaptic_Delay(delay_supra_basket, supra_basket_output)

        granular_ee = np.dot(granular_pyramidal_synout,Pee_granular)
        granular_ei = np.dot(granular_pyramidal_synout,np.transpose(Pie_granular))
        granular_ie = -np.dot(granular_basket_synout,Pie_granular)
        granular_ii = -np.dot(granular_basket_synout,Pii_granular)

        supra_ee = np.dot(supra_pyramidal_synout,Pee_supra)
        supra_ei = np.dot(supra_pyramidal_synout,np.transpose(Pie_supra))
        supra_ie = -np.dot(supra_basket_synout,Pie_supra)
        supra_ii = -np.dot(supra_basket_synout,Pii_supra)

        Layer_ei = np.reshape(granular_pyramidal_synout,(exc,exc))
        granular_supra_ee = np.dot(Layer_ei,Pee_vertical)
        granular_supra_ei = np.dot(np.transpose(Layer_ei[:,::2]),Pie_vertical)

        Input_granular_pyramidal1 = Input_granular_pyramidal - granular_ee - granular_ie
        Input_granular_basket1 = Input_granular_basket - granular_ei - granular_ii

        Input_supra_pyramidal = -supra_ee - supra_ie - np.reshape(granular_supra_ee,(n_exc,))
        Input_supra_basket = -supra_ii - supra_ei - np.reshape(granular_supra_ei,(n_inter,))
        
        P_granular_pyramidal.append(state_granular_pyramidal[:,0])
        P_granular_basket.append(state_granular_basket[:,0])
        P_supra_pyramidal.append(state_supra_pyramidal[:,0])
        P_supra_basket.append(state_supra_basket[:,0])

    return P_granular_pyramidal,P_granular_basket,P_supra_pyramidal,P_supra_basket

t0 = time.time()
main = dask.delayed(main)
m = main(24,48)
Potential = m.compute()
print(time.time()-t0)
