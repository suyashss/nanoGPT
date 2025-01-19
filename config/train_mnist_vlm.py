# train a miniature mnist vlm model

out_dir = 'out-mnist-vlm'
eval_interval = 500 
eval_iters = 2
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True 
wandb_project = 'tiny_vlm'
wandb_run_name = ""

dataset = 'mnist_vlm'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 32 # context of up to 32 previous characters

# baby GPT model :)
n_layer = 8
n_head = 8
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model
