import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import torch
import datetime, time
import io
import re
from csv import reader
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import glob, os
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from transformers import BertModel, AdamW, BertConfig,BertTokenizer
from train import Summarizer

import tensorflow as tf
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
from pyrouge import Rouge155

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    print('GPU device not found')


# In[83]:

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
BASE_DIR = "/home/rajivn/W266/W266-fall-2020-hwu-rnair/"

DAT_DIR = BASE_DIR+"data/ICSI_plus_NXT/tensors/"
RESULT_DIR = BASE_DIR+"/data/ICSI_plus_NXT/result/"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

files = glob.glob(RESULT_DIR+'*')
for f in files:
    os.remove(f)

train_d = dict()
val_d = dict()
test_d=dict()

for d_set in ("src", "labels", "segs", 'clss', "attn", "mask_cls"):
    train_d[d_set] = torch.load(DAT_DIR + d_set+"_"+"train.pt")
    val_d[d_set] = torch.load(DAT_DIR + d_set+"_"+"validation.pt")
    test_d[d_set] = torch.load(DAT_DIR + d_set+"_"+"test.pt")

train_dataset = TensorDataset(train_d["src"],train_d["labels"], train_d["segs"], 
                              train_d["clss"], train_d["attn"], train_d["mask_cls"])
val_dataset = TensorDataset(val_d["src"], val_d["labels"], val_d["segs"], 
                              val_d["clss"], val_d["attn"], val_d["mask_cls"])
test_dataset = TensorDataset(test_d["src"], test_d["labels"], test_d["segs"], 
                              test_d["clss"], test_d["attn"], test_d["mask_cls"])



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[84]:


# The DataLoader needs to know our batch size for training, so we specify it 
# here. For fine-tuning BERT on a specific task, the authors recommend a batch 
# size of 16 or 32.
batch_size = 4

# Create the DataLoaders for our training and validation sets.
# We'll take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order doesn't matter, so we'll just read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

model = Summarizer()



# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))


for param in model.bert.parameters():
    param.requires_grad=True

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )



from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# We chose to run for 4, but we'll see later that this may be over-fitting the
# training data.
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


def gen_outputs(batch_id, probs, labels, cls_ids, mask_cls,src):
    # extract sentences and labels
#    probs = np.where(probs>=0.12, 1, 0)
    probs_bin = np.where(probs>=0.12, 1, 0)
    reference = []
    result = []
    for p, passage in enumerate(src):
#        print(probs[p,])
        lines = tokenizer.decode(passage)
        lines = lines.split("[SEP]")
        for i, sent in enumerate(lines):
            sent = sent.replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "")
            with open(RESULT_DIR+"PRED"+batch_id+".txt", "a") as ref, open(RESULT_DIR+"LAB"+batch_id+".txt", "a") as lab, open(RESULT_DIR+"GOOD"+batch_id+".txt", "a") as good, open(RESULT_DIR+"BAD"+batch_id+".txt", "a") as bad:
                if labels[p, i] == 1:
#                    print("LABEL", "p=", p, "i=", i, probs[p, i], labels[p, i], sent)
                    reference.append(sent)
                    lab.write(sent + "\n")
                if probs_bin[p, i] == 1:
#                    print("PRED", "p=", p, "i=", i, probs[p, i], labels[p, i], sent)
                    result.append(sent)
                    ref.write(sent + "\n")
                if labels[p, i] == 1 and probs_bin[p, i] == 1:
                    good.write(str(probs[p, :np.argmin(mask_cls[p,])]) +"|"+ sent + "\n")
                if labels[p, i] != probs_bin[p, i]:
                    bad.write("perd = "+str(probs[p, :np.argmin(mask_cls[p,])]) +"|"+ "label = "+str(labels[p, i]) +"|"+sent + "\n")
    return reference, result # just return for debugging





import random
import numpy as np
import traceback


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
#torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()
loss_c = torch.nn.BCELoss(reduction='none')
# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        
        src, labels, segs, clss, attn, mask_cls = batch
        src, labels, segs, clss, attn, mask_cls = src.to(device), labels.to(device), segs.to(device), clss.to(device), attn.to(device), mask_cls.to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

#        x, segs, clss, mask, mask_cls, sentence_range=None
        probs, mask_cls = model( src, segs, clss, attn, mask_cls)
        loss = loss_c(probs, labels.float())
        loss = (loss * attn.float()).sum()

        probs = probs.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()


        gen_outputs("TRAIN"+str(step), probs, labels, clss, mask_cls, src)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = BASE_DIR+'data/model_save/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)


    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    torch.save(model.state_dict(), output_dir+"bertsum_classifier")



  # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    step = -1
#    for batch in validation_dataloader:
    for batch in train_dataloader:
        step += 1
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        src, labels, segs, clss, attn, mask_cls = batch
        src, labels, segs, clss, attn, mask_cls = src.to(device), labels.to(device), segs.to(device), clss.to(device), attn.to(device), mask_cls.to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        
            probs, mask_cls = model( src, segs, clss, attn, mask_cls)
            loss = loss_c(probs, labels.float())
            loss = (loss * attn.float()).sum()

            probs = probs.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        #print(type(logits), type(b_labels))
        
        probs = probs.numpy() #logits.detach().cpu().numpy()
        label_ids = labels.numpy() #b_labels.to('cpu').numpy()

        # write results so that we can use rouge to compare
        gen_outputs("BATCH"+str(step), probs, labels, clss, mask_cls, src)
        

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))