import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import matplotlib.pyplot as plt
from flax.linen.activation import softmax
from einops import rearrange
from jax import random, value_and_grad
from jax.example_libraries import optimizers
import optax
import sys
import wandb
import argparse
import os

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast
from itertools import chain



parser = argparse.ArgumentParser(description=''
    '''
    tests of multiple-layer-per-block models
    ''', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--mom', default=0.0, type=float, help='momentum')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--arch', type=str, default='LLM')
parser.add_argument('--dataset', type=str,default = 'C4')
parser.add_argument('--optimizer', default='adam')
parser.add_argument('--width', type=int, default = 32)
parser.add_argument('--heads', type=int, default = 4)
parser.add_argument('--patch_size',type=int, default=4)
parser.add_argument('--depth', type=int, default = 4)
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
    
if args.optimizer == "adam":
    adam = True
else:
    adam = False
    


save_dir = '/n/holyscratch01/pehlevan_lab/Users/bbordelon/bbordelon/Learn_gates/C4_LLM'


def get_run_name(args):
    return "model_{}/dataset_{}/optimizer_{}/lr_{:.4f}/batch_size_{}/steps_{}/width_{}/heads_{}/depth_{}/scale_exp_{}/beta_{}/gamma_zero_{}".format(args.arch, args.dataset, args.optimizer, args.lr, args.batch_size, args.steps, args.width, args.heads, args.depth, args.scale_exp, args.beta, args.gamma_zero)


# data stuff

MAX_LEN = args.max_len # length of the context window (consider making this longer)
BATCH_SIZE = args.batch_size # size of the batches


ds = load_dataset("allenai/c4", 'en', streaming = True)['train']
#ds = ds.with_format('jax')
shuff_ds = ds.shuffle(seed = 0, buffer_size = 10000)

tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

VOCAB_SIZE = len(tokenizer)


# encoder function: GPT byte encoding
def encode(examples):
    #return tokenizer(examples['text'], truncation=True, padding='max_length', max_length = MAX_LEN)
    return tokenizer(examples['text'])

# create groups of text of appropriate size
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    if total_length >= MAX_LEN:        
        total_length = (total_length // MAX_LEN) * MAX_LEN
    # Split by chunks of max_len.
    result = {
                k: [t[i : i + MAX_LEN] for i in range(0, total_length, MAX_LEN)]
                for k, t in concatenated_examples.items()
    }
    return result

dataset = shuff_ds.map(encode, batched= True, remove_columns=["timestamp", "url"])

#dataset_grouped = dataset.map(group_texts, batched = True, batch_size = BATCH_SIZE)
dataset_grouped = dataset.map(group_texts, batched = True)

import jax.numpy as jnp
from torch.utils.data import DataLoader
import numpy as np

# collate without mlm

collate_fn = lambda x: [ xi["input_ids"] for xi in x ] 

#dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, collate_fn = collate_fn)
dataloader = DataLoader(dataset_grouped, batch_size = BATCH_SIZE, collate_fn = collate_fn)


class Causal_Attention(nn.Module):

    scale_exp: jnp.float32
    dim: int
    heads: int
    qk_ln: bool = True
    
    def setup(self):
        
        self.c = 1.5 - self.scale_exp # exponent for the scale factor
        kif_qk = nn.initializers.normal(stddev = self.dim**(self.c - 0.5) ) # possibly needs to be scaled with N
        kif_v =  nn.initializers.normal(stddev = 1.0 ) # O_N(1) entries
        # computes key, query, value
        self.qk_layer = nn.Dense(features = 2 * self.heads * self.dim, kernel_init = kif_qk, use_bias = False)
        self.v_layer = nn.Dense(features = self.heads * self.dim, kernel_init = kif_v, use_bias = False)
        self.out_layer = nn.Dense(features = self.heads * self.dim, kernel_init = kif_v, use_bias = False)
        self.q_norm = nn.LayerNorm()
        self.k_norm = nn.LayerNorm()
        return
    
    def __call__(self,inputs):
        
        qk = self.qk_layer(inputs) / inputs.shape[-1]**(self.c)  # (batch, loc, 3*h*d)
        qk = rearrange( qk, 'b l (h d) -> b h l d' , h = self.heads) # (batch, heads, loc, d )
        q,k = jnp.split(qk, 2, axis = -1) # gives q, k each of shape ( batch, heads, loc, d )
        if self.qk_ln:
            q = self.q_norm( q )
            k = self.k_norm( k )
        
        v = self.v_layer(inputs) / jnp.sqrt( inputs.shape[-1] )
        v = rearrange(v, 'b l (h d) -> b h l d', h = self.heads)
        
        A = 1.0/ self.dim**(self.scale_exp) * jnp.einsum('ijkl,ijml->ijkm', q, k) # batch x heads x loc x loc
        exp_A =  jnp.einsum('ijkl,kl->ijkl', jnp.exp(A), jnp.tril(jnp.ones((v.shape[2], v.shape[2]))))
        phi_A = exp_A / exp_A.sum(axis = -1)[:,:,:,jnp.newaxis]  
        
        out = jnp.einsum('ijkl,ijlm->ijkm', phi_A, v) # (batch, head, loc, d)  
        out = rearrange(out, 'b h l d -> b l (h d)') 
        out = self.out_layer(out) / jnp.sqrt( out.shape[-1] )
        return out

    
class MLP_Block(nn.Module):

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
    d_model : int         # Hidden dimensionality of the input.
    scale: jnp.float32   # initial stddev of entries
    max_len : int = MAX_LEN  # Maximum length of a sequence to expect.
    
    def setup(self):
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        self.pos_embedding = self.param('pos_embedding', 
                                        nn.initializers.normal(stddev=self.scale), 
                                        (1, 1+self.max_len, self.d_model))

    def __call__(self, x, train=True):
        B,T,_ = x.shape
        x = x + self.pos_embedding[:,:T]
        return x

class Transformer(nn.Module):
    """A simple Decoder only transformer"""
  
    dim: int
    heads: int
    depth: int
    scale_exp: jnp.float32
    adam_scale: int
    beta: jnp.float32

    @nn.compact
    def __call__(self, x, train = True):
        N = self.heads * self.dim
        L = self.depth
        kif_first= nn.initializers.normal(stddev = N**(-0.5*self.adam_scale) * L**(0.5 * (1-self.adam_scale) ) ) # O_N(1) entries
        kif0 = nn.initializers.normal(stddev = 0.0 )
        kif = nn.initializers.normal(stddev = 1.0) # O_N(1) entries
        kif_last = nn.initializers.normal(stddev = L**(0.5 * (1-self.adam_scale)) * N**(-0.5*self.adam_scale) )
                
        # embed the batch x sequence integers to 
        x = L**( -0.5 * (1-self.adam_scale) )* N**(0.5 * self.adam_scale) * nn.Embed(VOCAB_SIZE, N, embedding_init = kif_first)(x) # batch x seq len x N
        x = N**(0.5 * self.adam_scale) * L**(-0.5 * (1-self.adam_scale) ) * PositionalEncoding(d_model = N, scale = N**(-0.5*self.adam_scale) * L**(0.5 * (1-self.adam_scale)) )(x)
        for l in range(self.depth):
            h = nn.LayerNorm()(x)
            x = x + self.beta/L * Causal_Attention(dim = self.dim, scale_exp = self.scale_exp, heads = self.heads)(nn.gelu(h))
            h = nn.LayerNorm()(x)
            x = x + self.beta/L * MLP_Block(features = N)(nn.gelu(h))
            
        x = nn.LayerNorm()(x)
        x = L**(-0.5 * (1 - self.adam_scale ) ) * nn.Dense(features = VOCAB_SIZE, use_bias = True, kernel_init = kif0)(x) / N**(1.0-0.5*self.adam_scale)   # for mean field scaling
        return x
    
    

def train_model(param_args, opt_args, data = None, adam = False):
    
    dim, heads, depth,scale_exp, beta = param_args
    lr , gamma, T = opt_args
   
    if adam:
        adam_scale = 1
        schedule = optax.warmup_cosine_decay_schedule(init_value=0.0,peak_value=lr / jnp.sqrt(heads*dim), warmup_steps=100,decay_steps=T,end_value=0.0)
        optimizer = optax.adamw( schedule , eps = 1e-20 , weight_decay = 0.0 )
    else:
        adam_scale = 0
        optimizer = optax.sgd( heads * dim *  lr)

    model = Transformer(dim, heads,depth, scale_exp = scale_exp, adam_scale = adam_scale, beta = beta)
    params = model.init(random.PRNGKey(0), jnp.ones((32,128), dtype = jnp.int32)) 
    loss_fn = jax.jit(lambda params, Xb, yb: optax.softmax_cross_entropy_with_integer_labels(logits=model.apply(params, Xb), labels=yb).mean())
    val_grad_fn = jax.jit(value_and_grad(loss_fn))


    opt_state = optimizer.init(params)

    run_loss = 0.0

    losses = []

    for t,batch in enumerate(dataloader):
        
        if np.amin(np.array([len(b) for b in batch])) == MAX_LEN:
        
            batch = jnp.array(batch)
    
            loss, grads = val_grad_fn(params, batch[:,:-1], batch[:,1:])
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            run_loss = loss
            #run_loss =  t/(t+1) * run_loss + 1/(t+1) * loss
            sys.stdout.write(f'\r loss = {run_loss}')
            wandb.log({'loss': run_loss})
            losses += [run_loss]
            if t > T:
                break
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
                opt_args = ( args.lr, args.gamma_zero, args.steps )
                param_args = (dim, head, depth, args.scale_exp , args.beta )

                wandb.init(
                    project="scaling_NL",
                    # track hyperparameters and run metadata
                    config=args.__dict__)
                
                wandb.run.name = run_name
                
                losses = train_model(param_args, opt_args, adam = adam)
                save_path = os.path.join(save_dir, run_name.replace("/", "-"))
                np.save(save_path, losses)
                wandb.finish()
