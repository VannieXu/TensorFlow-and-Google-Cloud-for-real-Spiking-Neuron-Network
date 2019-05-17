import numpy as np
import itertools
import scipy.spatial.distance

# Get a dask cluster
from dask.distributed import Client, config

config['scheduler-address']
client = Client()

# Get the ip addresses of each computing instance and add port to each ip
info = client.scheduler_info()
workers = iter(info['workers'])
addr = []
dask_addr = []
for i in range(len(info['workers'])):
    w = next(workers)
    d = w.split('://')[-1].rsplit(':')[0]
    addrs = '%s:%d'%(d,2222+i)
    dask_addr.append(w)
    addr.append(addrs)
 
# Define the computation graph as usual and use the ip:port address to create TensorFlow cluster
def work(addr,index):
    cluster = tf.train.ClusterSpec({"worker": addr})
    server = tf.train.Server(cluster, job_name="worker", task_index=index)
    #TensorFlow graph......
    
    with tf.Session(server.target) as sess:
    sess.run(tf.global_variables_initializer())
    if index == 0:
       # Computation in worker0
       
    elif index == 1:
       # Computation in worker1

worker_task = [client.submit(work,addr[i],i) for i in range(len(addr))]
worker0_result = worker_task[0].result()
worker1_result = worker_task[1].result()
