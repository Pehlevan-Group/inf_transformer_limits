import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import matplotlib.pyplot as plt
from flax.linen.activation import softmax
from einops import rearrange
from jax import random
from jax.example_libraries import optimizers
import optax
import sys
import wandb
import argparse
import os



parser = argparse.ArgumentParser(description=''
    '''
    tests of multiple-layer-per-block models
    ''', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--mom', default=0.0, type=float, help='momentum')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--arch', type=str, default='vit')
parser.add_argument('--dataset', type=str,default = 'cifar5m')
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--width', type=int, default = 32)
parser.add_argument('--heads', type=int, default = 4)
parser.add_argument('--patch_size',type=int, default=4)
parser.add_argument('--depth', type=int, default = 2)
parser.add_argument('--beta', type=float, default=4.0,
                         help='scaling factor for the residual branch. To use together with res_scaling parameter')
parser.add_argument('--gamma_zero', type=float, default=0.1,
                         help='controls the amount of feature learning.')
parser.add_argument('--scale_exp', type=float, default=1.0)
parser.add_argument('--steps',type=int, default = 2500)
parser.add_argument('--save_model', action='store_true')

args = parser.parse_args()

# set sweeps
if args.depth == -1:
    depths = [1,4,16,32]
else:
    depths = [args.depth]

if args.width == -1:
    widths = [16,32,64,128,256]
else:
    widths = [args.width]
    
if args.heads == -1:
    heads = [ 4,8,16,32,64 ]
else:
    heads = [args.heads]
    
if args.lr == -1:
    lrs = np.logspace(-2.5,1.0,12)
else:
    lrs = [args.lr]
    

save_dir = '/n/holyscratch01/pehlevan_lab/Users/bbordelon/bbordelon/Learn_gates/cifar_VIT'


def get_run_name(args):
    return "model_{}/dataset_{}/lr_{:.4f}/mom_{:.2f}/batch_size_{}/steps_{}/width_{}/heads_{}/depth_{}/scale_exp_{}/beta_{}/gamma_zero_{}".format(args.arch, args.dataset, args.lr, args.mom, args.batch_size, args.steps, args.width, args.heads, args.depth, args.scale_exp, args.beta, args.gamma_zero)



# MHSA attention layer
class Attention(nn.Module):

    scale_exp: jnp.float32
    dim: int
    heads: int
    
    def setup(self):
        
        self.c = 1.5 - self.scale_exp # exponent for the scale factor
        kif_qk = nn.initializers.normal(stddev = self.dim**(self.c - 0.5) ) # possibly needs to be scaled with N
        kif_v =  nn.initializers.normal(stddev = 1.0 ) # O_N(1) entries
        # computes key, query, value
        self.qk_layer = nn.Dense(features = 2 * self.heads * self.dim, kernel_init = kif_qk)
        self.v_layer = nn.Dense(features = self.heads * self.dim, kernel_init = kif_v)
        return
    
    def __call__(self,inputs):
        
        qk = self.qk_layer(inputs) / inputs.shape[-1]**(self.c)  # (batch, loc, 3*h*d)
        qk = rearrange( qk, 'b l (h d) -> b h l d' , h = self.heads) # (batch, heads, loc, d )
        q,k = jnp.split(qk, 2, axis = -1) # gives q, k each of shape ( batch, heads, loc, d )
        v = self.v_layer(inputs) / jnp.sqrt( inputs.shape[-1] )
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.heads)
        A = 1.0/ self.dim**(self.scale_exp) * jnp.einsum('ijkl,ijml->ijkm', q, k) # batch x heads x loc x loc
        phi_A = softmax( A, axis=-1 )
        out = jnp.einsum('ijkl,ijlm->ijkm', phi_A, v) # (batch, head, loc, d)  
        out = rearrange(out, 'b h l d -> b l (h d)')
        return out
    
class MLP_Block(nn.Module):

    features: int
    
    @nn.compact
    def __call__(self,x):
        N = self.features
        kif = nn.initializers.normal(stddev = 1.0) # O_N(1) entries
        h = nn.Dense(features = 2*N, kernel_init = kif)(x) / jnp.sqrt(N)
        h = nn.relu(h)
        h = nn.Dense(features = N, kernel_init = kif)(x) / jnp.sqrt(2*N)
        return h
    

class VIT(nn.Module):
    """A simple VIT model"""
  
    dim: int
    heads: int
    depth: int
    patch_size: int
    scale_exp: jnp.float32
    adam_scale: int
    beta: jnp.float32

    @nn.compact
    def __call__(self, x, train = True):
        N = self.heads * self.dim
        D = 3
        
        
        # patchify images
        x = rearrange(x, 'b (w p1) (h p2) c -> b (w h) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size) # (batch, loc, patch_ch_dim )
        
        kif_first= nn.initializers.normal(stddev = N**(-0.5*self.adam_scale) ) # O_N(1) entries
        kif0 = nn.initializers.normal(stddev = 0.0 )
        kif = nn.initializers.normal(stddev = 1.0) # O_N(1) entries

        x = N**(0.5 * self.adam_scale) * nn.Dense(features = N, kernel_init = kif_first)(x) / jnp.sqrt( 3 * self.patch_size**2 )
        x = nn.relu(x)
        for l in range(self.depth):
            h = nn.LayerNorm()(x)
            x = x + self.beta/jnp.sqrt(depth) * Attention(dim = self.dim, scale_exp = self.scale_exp, heads = self.heads)(h)
            h = nn.LayerNorm()(x)
            x = x + self.beta/jnp.sqrt(depth) * MLP_Block(features = N)(h)
            
        # pool over location index
        x = x.mean(axis = 1) # (batch, N)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features = 10, use_bias = False, kernel_init = kif0)(x) / N**(1.0-0.5*self.adam_scale)   # for mean field scaling
        return x
    

    

data_dir = '/n/holyscratch01/pehlevan_lab/Everyone/cifar-5m-new'
file_name = f"{data_dir}/cifar5m_part{0}.npz"
part0 = np.load(file_name, allow_pickle=True)
X0, y0 = np.load(file_name, allow_pickle=True)

arr = [part0[k] for k in part0.keys()]
X = arr[0]
y = arr[1]
mean = X.mean()
std = X.std()
X = (X - mean) / std
print(y.shape)
print(X.shape)


def get_data(dset_count):
    
    data_dir = '/n/holyscratch01/pehlevan_lab/Everyone/cifar-5m-new'
    file_name = f"{data_dir}/cifar5m_part{dset_count}.npz"
    part0 = np.load(file_name, allow_pickle=True)

    arr = [part0[k] for k in part0.keys()]
    X = arr[0]
    y = arr[1]
    mean = X.mean()
    std = X.std()
    X = (X - mean) / std
    return X, y

def train_model(param_args, opt_args, data = None, adam = False):

    dim, heads, depth, patch_size, scale_exp, beta = param_args
    T, batch, gamma, lr = opt_args
    
    if adam:
        adam_scale = 1.0
        opt_init, opt_update, get_params = optimizers.adam( lr / jnp.sqrt(heads * dim) , eps = 1e-20)

    else:
        adam_scale = 0.0
        opt_init, opt_update, get_params = optimizers.sgd( heads * dim * gamma**2 *  lr)

    model = VIT(dim = dim, heads = heads, depth = depth, patch_size = patch_size, scale_exp = scale_exp, adam_scale = adam_scale, beta = beta)
    
    params = model.init(random.PRNGKey(0), jnp.ones((4,32,32,3)) )['params']                                                               
    opt_state = opt_init(params)
    
    shift_fn = jax.jit(lambda p, x: (model.apply({'params':p}, x) - model.apply({'params':params}, x)) / gamma)
    #shift_fn = jax.jit(lambda p, x: model.apply({'params':p}, x) / gamma)
    loss_fn = jax.jit(lambda params, Xb, yb: optax.softmax_cross_entropy_with_integer_labels(logits=shift_fn(params, Xb), labels=yb).mean())
    grad_fn = jax.jit(jax.grad(loss_fn))


    losses = []
    
    loss_t = 0.0
    dset_count = 0
    
    if data != None:
        X,y = data
    
    for t in range(T):
    
        if data == None:
            if t == 0:
                X, y = get_data(dset_count)
                ind = 0
            elif ind+batch >= X.shape[0]:
            
                if dset_count < 4:
                    dset_count += 1
                    X, y = get_data(dset_count)
                    ind = 0
                else:
                    return losses
        else:
            ind = (t*batch) % X.shape[0]
        Xt = X[ind:ind+batch]
        yt = y[ind:ind+batch]
        ind += batch
        
        loss_t = loss_t * t/(t+1) + 1.0/(t+1) * loss_fn(get_params(opt_state), Xt, yt)
        losses += [loss_t]
        if t % 10 == 0:
            sys.stdout.write(f'\r loss = {loss_t}')
            wandb.log({'loss': loss_t })
            
        opt_state = opt_update(t,grad_fn(get_params(opt_state), Xt, yt),opt_state)
    
    return losses


# sweep over width, depth, gates, and lrs
for i, dim in enumerate(widths):
    for j, depth in enumerate(depths):
        for k, head in enumerate(heads):
            for l, lr in enumerate(lrs):
            
                args.width = dim
                args.depth = depth
                args.lr = lr
                args.heads = head
                
                run_name = get_run_name(args)
                
                    
                opt_args = ( args.steps , args.batch_size, args.gamma_zero, args.lr )
                param_args = (dim, head, depth, args.patch_size, args.scale_exp , args.beta )

                wandb.init(
                    project="scaling_NL",
                    # track hyperparameters and run metadata
                    config=args.__dict__)
                
                wandb.run.name = run_name
                losses = train_model(param_args, opt_args, (X, y))
                save_path = os.path.join(save_dir, run_name.replace("/", "-"))
                np.save(save_path, losses)
                wandb.finish()