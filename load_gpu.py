# stolen from Ilya (/u/ilya/py/gpu.py)
import gpu_lock
lock_id = gpu_lock.obtain_lock_id()
import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu%s,floatX=float32' % \
    lock_id
import theano

