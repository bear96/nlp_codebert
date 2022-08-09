import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class InputFeatures(object):
  #to process data into ids and mask for the custom dataloader
  def __init__(self,input_ids,input_mask,target_ids,target_mask):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.target_ids =target_ids
    self.target_mask = target_mask

        
def convert_examples_to_features(dataset,tokenizer,block_size):
    #target
    exps = dataset['expansions']
    exps_tokens=tokenizer.tokenize(exps)[:block_size-2]
    target_tokens =[tokenizer.cls_token]+exps_tokens+[tokenizer.sep_token]
    target_ids =  tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] *len(target_ids)
    padding_length = block_size - len(target_ids)
    target_ids+=[tokenizer.pad_token_id]*padding_length
    target_mask+=[0]*padding_length   
    
    #source
    facts = dataset['factors']
    facts_tokens=tokenizer.tokenize(facts)[:block_size-2]
    source_tokens =[tokenizer.cls_token]+facts_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask = [1] * (len(source_tokens))
    source_mask+=[0]*padding_length

    return InputFeatures(source_ids,source_mask,target_ids,target_mask)

class NLPData(Dataset):
    def __init__(self, tokenizer, dataset,blocksize=30):
        self.examples = []
        for i in tqdm(range(len(dataset)),desc = "Processing dataset..."):
          x = dataset.iloc[i]
          self.examples.append(convert_examples_to_features(x,tokenizer,blocksize))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, indx):       
        return torch.tensor(self.examples[indx].input_ids),torch.tensor(self.examples[indx].input_mask),torch.tensor(self.examples[indx].target_ids),torch.tensor(self.examples[indx].target_mask)
