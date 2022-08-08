import pandas as pd
import numpy as np 
from typing import Tuple
import sys
import argeparse
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

def train(train_dataset,val_dataset,args):

    config= RobertaConfig.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base',config=config)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device: ",device)

    model = NLPTransformer(tokenizer= tokenizer, config=config,beam_size= 10,max_length=args.blocksize,
                           sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id).to(device)

    torch.cuda.empty_cache()

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=2)
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle = False,num_workers=2)
    epochs= args.epoch

    optimizer_grouped_parameters = [{'params': [p for n, p in model.named_parameters()],}]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr= 0.00005)
    
    train_loss_graph = []
    val_loss_graph = []
    for idx in range(epochs): 
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
        print("Epoch: {0:.4f} Train loss: {0:.4f} Val loss: {0:.4f}".format(idx+1,epoch_loss,epoch_val_loss))
        train_loss_graph.append(epoch_loss)
        val_loss_graph.append(epoch_val_loss)
        torch.save(model.state_dict(), 'CodeBERTmodel-nlp-{}.pkl'.format(idx+1))
        if epoch_val_loss>epoch_loss:
            print("Model is overfitting...\nStopping training")
            break
            
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.plot(train_loss_graph,'k')
    plt.plot(val_loss_graph,'y')
    plt.legend(["Training Loss","Validation Loss"])
    plt.savefig('Plot-Loss-nlp-CodeBERT.png')

def predict(test_dataset,args):
    config= RobertaConfig.from_pretrained('microsoft/codebert-base')
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base',config=config)
    model = NLPTransformer(tokenizer= tokenizer, config=config,beam_size= 10,max_length=args.blocksize,
                sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id).to(device)

    model.load_state_dict(torch.load('CodeBERTmodel-nlp-{}.pkl'.format(args.epoch)))

    test_dataset = process_dataset.NLPData(tokenizer, test_data[:1024])

    batch_size = 1
    torch.cuda.empty_cache()
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,num_workers=1)
    preds = []
    for batch in tqdm(test_dataloader):
        source_ids = batch[0].to(device) 
        source_mask = batch[1].to(device)
        model.eval()
        output = model(source_ids=source_ids,source_mask=source_mask)
        preds.append(output)

    prediction = []
    for line in preds:
        temp = []
        for i in range(10):
            code = line[0][i]
            gen_code = tokenizer.decode(code,skip_special_tokens=True)
            temp.append(gen_code)
        prediction.append(temp)

    predictions = pd.DataFrame({'predicted_expansions': prediction,'actual_expansions': test_data[:1024].expansions})

    predictions.to_json('Predicted_Expansions_CodeBERT.json')
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='train.txt', type=str, help="Directory where the dataset is.")
    parser.add_argument("--blocksize", default = 30, type = int, help = "Maximum length of sequence. Default = 30")
    parser.add_argument("--batch_size", default = 32, type = int, help = "Batch size for training and validation. Default = 32")
    parser.add_argument("--epoch", default = 10, type = int, help = "Number of epochs for training. Default = 10.")
    parser.add_argument("--do_train", type = bool, required = True, help = "True if doing training, False if doing testing.")
    
    args = parser.parse_args()
    
    
    factors, expansions = load_file(args.data_dir)
    dataset = pd.DataFrame({'factors':factors,'expansions':expansions})
    print("First five factors-expansions pair")
    print(dataset.head())
    
    train_data, val_data = model_selection.train_test_split(dataset[:10000],train_size=0.7)
    val_data, test_data = model_selection.train_test_split(val_data,train_size=0.5)
    
    print("Training data...")
    train_dataset = process_dataset.NLPData(tokenizer, train_data)
    print("Validation data...")
    val_dataset = process_dataset.NLPData(tokenizer,val_data)
    
    if args.do_train:
        train(train_dataset,val_dataset,args)
    else:
        predict(test_data,args)
       
