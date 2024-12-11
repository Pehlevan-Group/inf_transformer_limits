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
from flax.training import orbax_utils
from flax.training import checkpoints, train_state
import orbax.checkpoint


parser = argparse.ArgumentParser(description=''
    '''
    tests of multiple-layer-per-block models
    ''', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='momentum')
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
parser.add_argument('--depth_exp', type=float, default=1.0)
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
elif args.lr == -2:
    lrs = np.logspace(-3, 0.0, 12)
else:
    lrs = [args.lr]

depth_exp = args.depth_exp
beta = args.beta

save_dir = '/n/holyscratch01/pehlevan_lab/Users/bbordelon/bbordelon/Learn_gates/cifar_VIT'


def get_run_name(args):
    return "model_{}/dataset_{}/lr_{:.4f}/mom_{:.2f}/batch_size_{}/steps_{}/width_{}/heads_{}/depth_{}/scale_exp_{}/depth_exp_{}/beta_{}/gamma_zero_{}".format(args.arch, args.dataset, args.lr, args.mom, args.batch_size, args.steps, args.width, args.heads, args.depth, args.scale_exp, args.depth_exp, args.beta, args.gamma_zero)


class LN_Fixed(nn.Module):
    
    eps: jnp.float32 = 1.0e-6
    @nn.compact
    
    def __call__(self, x):
        
        features = x.shape[-1] # number of features
        
        mean = jnp.mean( x , axis = -1 ) # mean of x
        var = jnp.var( x , axis = -1 )
        out = (x - mean[:,:,jnp.newaxis] ) / jnp.sqrt( var[:,:,jnp.newaxis] + self.eps )
        return out

# MHSA attention layer
class Attention(nn.Module):
    """Multi-head Self-Attention Layer"""
    scale_exp: jnp.float32
    dim: int
    heads: int
    qk_layernorm: bool = False
    
    def setup(self):
        
        self.c = 1.5 - self.scale_exp # exponent for the scale factor
        kif_qk = nn.initializers.normal(stddev = self.dim**(self.c - 0.5) ) # possibly needs to be scaled with N
        kif_v =  nn.initializers.normal(stddev = 1.0 ) # O_N(1) entries
        # computes key, query, value
        self.qk_layer = nn.Dense(features = 2 * self.heads * self.dim, kernel_init = kif_qk, use_bias = False)
        self.v_layer = nn.Dense(features = self.heads * self.dim, kernel_init = kif_v, use_bias = False)
        self.out_layer = nn.Dense(features = self.heads * self.dim, kernel_init = kif_v, use_bias = False)
        #self.norm_q = nn.LayerNorm()
        #self.norm_k = nn.LayerNorm()
        self.norm_q = LN_Fixed()
        self.norm_k = LN_Fixed()
        return
    
    def __call__(self,inputs):
        
        qk = self.qk_layer(inputs) / self.heads**(0.5) / self.dim**(self.c) / jnp.sqrt(2.0)
        qk = rearrange( qk, 'b l (h d) -> b h l d' , h = self.heads) # (batch, heads, loc, d )
        q,k = jnp.split(qk, 2, axis = -1) # gives q, k each of shape ( batch, heads, loc, d )
        
        if self.qk_layernorm: 
            q = self.norm_q(q)
            k = self.norm_k(k)
            
        v = self.v_layer(inputs) / jnp.sqrt( inputs.shape[-1] )
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.heads)
        A = 1.0/ self.dim**(self.scale_exp) * jnp.einsum('ijkl,ijml->ijkm', q, k) # batch x heads x loc x loc
        phi_A = softmax( A, axis=-1 )
        out = jnp.einsum('ijkl,ijlm->ijkm', phi_A, v) # (batch, head, loc, d)  
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.out_layer(out) / jnp.sqrt( out.shape[-1] )
        return out
    
class MLP_Block(nn.Module):
    """Two Layer MLP Block"""
    features: int
    
    @nn.compact
    def __call__(self,x):
        N = self.features
        kif = nn.initializers.normal(stddev = 1.0) # O_N(1) entries
        h = nn.Dense(features = N, kernel_init = kif, use_bias = False)(x) / jnp.sqrt(N)
        h = nn.gelu(h)
        h = nn.Dense(features = N, kernel_init = kif, use_bias = False)(h) / jnp.sqrt(N)
        return h

class PositionalEncoding(nn.Module):
    """Trainable Positional Encoding"""
    d_model : int         # Hidden dimensionality of the input.
    max_len : int  # Maximum length of a sequence to expect.
    scale: jnp.float32 # scale parameter for initialization
    
    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        self.pos_embedding = self.param('pos_embedding', 
                                        nn.initializers.normal(stddev = self.scale), 
                                        (1, 1+self.max_len, self.d_model))

    def __call__(self, x, train=True):
        B,T,_ = x.shape
        x = x + self.pos_embedding[:,:T] / self.scale
        return x

class simple_TF(nn.Module):
    
    """Simple TF Like Model, Easy to Analyze"""
    dim: int
    heads: int
    depth: int
    patch_size: int
    scale_exp: jnp.float32 = 1.0
    adam_scale: int = 0.0
    beta: jnp.float32 = 6.0
    
    @nn.compact
    def __call__(self, x, train = True):
        N = self.heads * self.dim
        L = self.depth
        D = 3
        
        # patchify images
        x = rearrange(x, 'b (w p1) (h p2) c -> b (w h) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size) # (batch, loc, patch_ch_dim )
        
        kif_first= nn.initializers.normal(stddev = N**(-0.5*self.adam_scale) * (L/self.beta)**(0.5 * (1.0-self.adam_scale)) ) # O_N(1) entries
        kif = nn.initializers.normal( stddev = 1.0 ) # O_N(1) entries
        kif_last = nn.initializers.normal(stddev = (L/self.beta)**(0.5 * (1-self.adam_scale) ) )
        
        x = (L/self.beta)**(-0.5 * (1.0-self.adam_scale)) * N**(0.5 * self.adam_scale) * nn.Dense(features = N, kernel_init = kif_first, use_bias = False)(x) / jnp.sqrt( D * self.patch_size**2 )
        for l in range(self.depth):
            h = Attention(dim = self.dim, scale_exp = self.scale_exp, heads = self.heads)( nn.gelu(x) ) 
            h = nn.Dense(features = N, kernel_init = kif, use_bias = False)( nn.gelu(h) ) / jnp.sqrt(N)
            x = x + self.beta / L * h
            
        # pool over location index
        x = x.mean(axis = 1) # (batch, N)
        x = (L/self.beta)**(-0.5*(1-self.adam_scale)) * nn.Dense(features = 10, use_bias = False, kernel_init = kif_last)(x) / N**(1.0-0.5*self.adam_scale)   # for mean field scaling
        return x


class VIT(nn.Module):
    
    "simple VIT model with "
    dim: int
    heads: int
    depth: int
    patch_size: int
    scale_exp: jnp.float32 = 1.0
    depth_exp: jnp.float32=1.0
    adam_scale: int = 0.0
    beta: jnp.float32 = 4.0
    L0: jnp.float32 = 2.0
    
    @nn.compact
    def __call__(self, x, train = True):
        N = self.heads * self.dim
        L = self.depth
        D = 3
        
        # patchify images
        x = rearrange(x, 'b (w p1) (h p2) c -> b (w h) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size) # (batch, loc, patch_ch_dim )
        
        L0 = self.L0
        
        L_scale_factor = (L / L0)**(-0.5 * (1.0-self.adam_scale) * ( 2*self.depth_exp - 1.0 ) )

        kif_first= nn.initializers.normal(stddev = N**(-0.5*self.adam_scale) / L_scale_factor ) # O_N(1) entries
        kif = nn.initializers.normal( stddev = 1.0 ) # O_N(1) entries
        kif_last = nn.initializers.normal(stddev = 1.0 / L_scale_factor )
        
        
        # read-in weights
        x = L_scale_factor * N**(0.5 * self.adam_scale) * nn.Dense(features = N, kernel_init = kif_first, use_bias = False)(x) / jnp.sqrt( D * self.patch_size**2 )
        # positional encoding
        #x = PositionalEncoding(d_model = N, max_len = (32//self.patch_size)**2, scale = N**(-0.5*self.adam_scale) / L_scale_factor)(x)
        # residual stream with pre-LN
        for l in range(self.depth):
            h = LN_Fixed()(x)
            h = Attention(dim = self.dim, scale_exp = self.scale_exp, heads = self.heads )( h )
            x = x + self.beta * (L / L0)**(-self.depth_exp) * h
            h = LN_Fixed()(x)
            h = MLP_Block(features = N)(h)
            x = x + self.beta * (L / L0)**(-self.depth_exp) * h
        
        # last norm layer
        x = LN_Fixed()(x)
        # pool over spatial dimension
        x = x.mean(axis = 1) # (batch, N)
        x = L_scale_factor * nn.Dense(features = 10, use_bias = False, kernel_init = kif_last)(x) / N**(1.0-0.5*self.adam_scale)   # for mean field scaling
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

    dim, heads, depth, patch_size, scale_exp, depth_exp, beta = param_args
    T, batch, gamma, lr, mom = opt_args
    
    L0 = 2.0
    
    if adam:
        adam_scale = 1.0
        opt_init, opt_update, get_params = optimizers.adam( lr / jnp.sqrt(heads * dim) , eps = 1e-20)

    else:
        adam_scale = 0.0
        opt_init, opt_update, get_params = optimizers.momentum( heads * dim * gamma**2 * (depth / L0)**(2 * depth_exp - 1.0) *  lr, mom)

    if args.arch == "vit":
        model = VIT(dim = dim, heads = heads, depth = depth, patch_size = patch_size, scale_exp = scale_exp, depth_exp = depth_exp, adam_scale = adam_scale, beta = beta)
    else:
        model = simple_TF(dim = dim, heads = heads, depth = depth, patch_size = patch_size, scale_exp = scale_exp, depth_exp=depth_exp, adam_scale = adam_scale, beta = beta)

    
    params = model.init(random.PRNGKey(0), jnp.ones((4,32,32,3)) )['params']                                                               
    opt_state = opt_init(params)
    
    shift_fn = jax.jit(lambda p, x: (model.apply({'params':p}, x) - model.apply({'params':params}, x)) / gamma)
    #shift_fn = jax.jit(lambda p, x: model.apply({'params':p}, x) / gamma)
    loss_fn = jax.jit(lambda params, Xb, yb: optax.softmax_cross_entropy_with_integer_labels(logits=shift_fn(params, Xb), labels=yb).mean())
    grad_fn = jax.jit(jax.grad(loss_fn))


    losses = []
    
    loss_t = 0.0
    dset_count = 0
    comp_every = 25
    
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
                else:
                    dset_count = 0
                    
                X, y = get_data(dset_count)
                ind = 0
        else:
            ind = (t*batch) % X.shape[0]
        Xt = X[ind:ind+batch]
        yt = y[ind:ind+batch]
        ind += batch
        
        if t % comp_every == 0 and t > 0:
            losses += [loss_t]
            wandb.log({'loss': loss_t })
            sys.stdout.write(f'\r loss = {loss_t}')
            loss_t = 0.0

        loss_t += 1/comp_every * loss_fn(get_params(opt_state), Xt, yt)
            
        opt_state = opt_update(t,grad_fn(get_params(opt_state), Xt, yt),opt_state)
    
    return losses, get_params(opt_state), model


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
                    
                opt_args = ( args.steps , args.batch_size, args.gamma_zero, args.lr, args.mom )
                param_args = (dim, head, depth, args.patch_size, args.scale_exp , args.depth_exp , args.beta )

                wandb.init(
                    project="scaling_NL",
                    # track hyperparameters and run metadata
                    config=args.__dict__)
                
                wandb.run.name = run_name
                losses, params, model = train_model(param_args, opt_args, data = None)
                save_path = os.path.join(save_dir, run_name.replace("/", "-"))
                np.save(save_path, losses)
                
                
                
                if args.save_model:
                    state = train_state.TrainState.create(
                        apply_fn=model.apply,
                        params=params,
                        tx= optax.sgd(lr))
                    
                    ckpt = {'model': state, 'config': args, 'losses': losses}
                    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                    save_args = orbax_utils.save_args_from_target(ckpt)
                    orbax_checkpointer.save(save_path + '_ckpts', ckpt, save_args=save_args)

                
                wandb.finish()