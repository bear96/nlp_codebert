import pandas as pd
import numpy as np 
from typing import Tuple
import sys
import argparse
from transformers import RobertaTokenizer,T5ForConditionalGeneration
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
#from torchinfo import summary

from sklearn import metrics, model_selection
from matplotlib import pyplot as plt 
from tqdm import tqdm 
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

def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)
  
def train(train_data,val_data,args):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Device: ",device)
    
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base').to(device)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    
    torch.cuda.empty_cache()
    print("Training data...")
    train_dataset = process_dataset.NLPData(tokenizer, train_data)
    print("Validation data...")
    val_dataset = process_dataset.NLPData(tokenizer,val_data)

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=1)
    val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,shuffle = False,num_workers=1)
    epochs= args.epoch

    optimizer = optim.AdamW(model.parameters(), lr= args.lr)
    
    train_loss_graph = []
    val_loss_graph = []
    for idx in range(epochs):
      tr_loss = 0.0
      val_loss = 0.0
      
      for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch[0].to(device) #input IDs
        target_ids = batch[2].to(device) #targets
        source_mask = batch[1].to(device)
        target_mask = batch[3].to(device)

        model.train()

        model_out = model(input_ids=input_ids, attention_mask=source_mask,labels=target_ids, decoder_attention_mask=target_mask)
        loss = model_out.loss
        loss.backward() #backpropagation
        optimizer.step()
        tr_loss += loss.item()
      epoch_loss = tr_loss/len(train_dataloader) #training loss per epoch

      #validation
      for batch in val_dataloader:
        model.eval()

        input_ids = batch[0].to(device) #input IDs
        target_ids = batch[2].to(device) #targets
        source_mask = batch[1].to(device)
        target_mask = batch[3].to(device)
        with torch.no_grad():
          model_out = model(input_ids=input_ids, attention_mask=source_mask,labels=target_ids, decoder_attention_mask=target_mask)
          v_loss = model_out.loss
          val_loss +=v_loss.item()
        epoch_val_loss = val_loss/len(val_dataloader) #validation loss per epoch
      print("epoch {} train loss {} val loss {}".format(idx+1,epoch_loss,epoch_val_loss))
      train_loss_graph.append(epoch_loss) 
      val_loss_graph.append(epoch_val_loss)
      torch.save(model.state_dict(), 'CodeT5model-{}.pkl'.format(idx+1))
      #print("Epoch: {0:.4f} Train loss: {0:.4f} Val loss: {0:.4f}".format(idx+1,epoch_loss,epoch_val_loss))
      train_loss_graph.append(epoch_loss)
      val_loss_graph.append(epoch_val_loss)
      #torch.save(NLPmodel.state_dict(), 'CodeBERTmodel-nlp-{}.pkl'.format(idx+1))
      if epoch_val_loss>epoch_loss:
        print("Model is overfitting...\nStopping training")
        break
            
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.plot(train_loss_graph,'k')
    plt.plot(val_loss_graph,'y')
    plt.legend(["Training Loss","Validation Loss"])
    plt.savefig('Plot-Loss-nlp-CodeT5.png')

def predict(test_data,args):
    if torch.cuda.is_available:
        device = 'cuda'
    else:
        device = 'cpu'
    
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base').to(device)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    model.load_state_dict(torch.load('CodeT5model-{}.pkl'.format(args.stopping_no)))

    print("Testing data...")
    test_dataset = process_dataset.NLPData(tokenizer, test_data)
    
    torch.cuda.empty_cache()
    test_dataloader = DataLoader(test_dataset,batch_size=1,num_workers=1)
    preds = []
    for batch in tqdm(test_dataloader):
        input_ids = batch[0].to(device)
        model.eval()
        output = model.generate(input_ids.reshape(1,-1), max_length=30)
        expans = tokenizer.decode(output[0],skip_special_tokens=True)
        preds.append(expans)
        

    predictions = pd.DataFrame({'predicted_expansions': preds,'actual_expansions': test_data.expansions})
    predictions.to_json('Predicted_Expansions_CodeT5.json')
    
    count = 0
    for i in range(len(preds)):
        count += score(preds[i],test_data.expansions.iloc[i])
    
    print("Accuracy: ",count/len(preds))

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='train.txt', type=str, help="Directory where the dataset is.")
    parser.add_argument("--blocksize", default = 30, type = int, help = "Maximum length of sequence. Default = 30")
    parser.add_argument("--batch_size", default = 32, type = int, help = "Batch size for training and validation. Default = 32")
    parser.add_argument("--epoch", default = 10, type = int, help = "Number of epochs for training. Default = 10.")
    parser.add_argument("--do_train",action = "store_true", help = "Specify if training, unspecified means testing.")
    parser.add_argument("--stopping_no", default = 10, type = int, help = "Epoch number where the model stopped training. Used to load last saved model for testing.")
    parser.add_argument("--lr", default = 5e-5, type = float, help = "Learning rate of optimizer. Default = 0.00005")
    parser.add_argument("--test_dir", default = "", type = str, help = "Directory/filename of new test file.")
    
    args = parser.parse_args()
    print(args)
    
    factors, expansions = load_file(args.data_dir)
    dataset = pd.DataFrame({'factors':factors,'expansions':expansions})
    
    train_data, val_data = model_selection.train_test_split(dataset,train_size=0.7)
    val_data, test_data = model_selection.train_test_split(val_data,train_size=0.5)
    

    
    if args.do_train == True:
        train(train_data,val_data,args)
    elif args.do_train == False and args.test_dir =="":
        predict(test_data,args)
    else:
        factors, expansions = load_file(args.test_dir)
        test_data = pd.DataFrame({'factors':factors,'expansions':expansions})
        predict(test_data,args)
       
if __name__ == "__main__":
    main()
