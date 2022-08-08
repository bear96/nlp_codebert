import pandas as pd
import numpy as np 
from typing import Tuple
import sys

from transformers import RobertaTokenizer,RobertaConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from sklearn import metrics, model_selection
from matplotlib import pyplot as plt 
from tqdm import tqdm 
import gc
from model import NLPTransformer
import beam
import process_dataset


def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions

factors, expansions = load_file("train.txt")
dataset = pd.DataFrame({'factors':factors,'expansions':expansions})
print("First five factors-expansions pair")
print(dataset.head())

train_data, val_data = model_selection.train_test_split(dataset[:10000],train_size=0.6)
val_data, test_data = model_selection.train_test_split(val_data,train_size=0.5)

blocksize = 30
config= RobertaConfig.from_pretrained('microsoft/codebert-base')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base',config=config)

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print("Device: ",device)

model = NLPTransformer(tokenizer= tokenizer, config=config,beam_size= 10,max_length=blocksize,sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id).to(device)

train_batch_size = 32
torch.cuda.empty_cache()
print("Training data...")
train_dataset = process_dataset.NLPData(tokenizer, train_data[:5000])
print("Validation data...")
val_dataset = process_dataset.NLPData(tokenizer,val_data[:500])

train_dataloader = DataLoader(train_dataset, 
                                  batch_size=train_batch_size,shuffle=True,num_workers=2)
val_dataloader = DataLoader(val_dataset, 
                                  batch_size=train_batch_size,shuffle = False,num_workers=2)
num_train_epochs= 5

optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters()],}]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr= 0.00005)
    
train_loss_graph = []
val_loss_graph = []
for idx in range(num_train_epochs): 

  tr_loss = 0.0
  val_loss = 0.0
        
  for batch in tqdm(train_dataloader):
    optimizer.zero_grad()
    source_ids = batch[0].to(device) 
    source_mask = batch[1].to(device)
    target_ids = batch[2].to(device)
    target_mask = batch[3].to(device)#target_ids.ne(tokenizer.pad_token_id)
    labels=batch[1].to(device)
    model.train()
    loss,_,_ = model(source_ids=source_ids,source_mask=source_mask.float(),target_ids=target_ids,target_mask=target_mask.float())
    loss.backward()
    optimizer.step()
    tr_loss += loss.item()

  epoch_loss = tr_loss/len(train_dataloader)

  for batch in val_dataloader:
    source_ids = batch[0].to(device) 
    source_mask = batch[1].to(device)
    target_ids = batch[2].to(device)
    target_mask = batch[3].to(device)
    labels=batch[1].to(device)
    model.eval()
    with torch.no_grad():
      v_loss,_,_ = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)
      val_loss +=v_loss.item()
  epoch_val_loss = val_loss/len(val_dataloader)
  print("epoch {} train loss {} val loss {}".format(idx+1,epoch_loss,epoch_val_loss))
  train_loss_graph.append(epoch_loss)
  val_loss_graph.append(epoch_val_loss)
  torch.save(model.state_dict(), 'CodeBERTmodel-PY-{}.pkl'.format(idx+1))

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.plot(train_loss_graph,'k')
plt.plot(val_loss_graph,'y')
plt.legend(["Training Loss","Validation Loss"])
plt.savefig('Plot-Loss-PY-CodeBERT.png')

d = pd.DataFrame({'train_loss':train_loss_graph,'val_loss':val_loss_graph})
d.to_csv('Losses_for_PY_CodeBERT.csv')
