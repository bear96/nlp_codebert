import pandas as pd
import numpy as np
from scipy import stats 
from typing import Tuple
import sys
from transformers import RobertaTokenizer,RobertaModel,pipeline,RobertaConfig, AdamW
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn import metrics, model_selection
from matplotlib import pyplot as plt 
from tqdm import tqdm 
import gc
from sklearn import metrics, model_selection

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
dataset.head()

train_data, val_data = model_selection.train_test_split(dataset,train_size=0.6)
val_data, test_data = model_selection.train_test_split(val_data,train_size=0.5)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_ids,
                 input_mask,
                 target_ids,
                 target_mask

    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.target_ids =target_ids
        self.target_mask = target_mask

        
def convert_examples_to_features(js,tokenizer,block_size):
    #source
    code = js['expansions']
    code_tokens=tokenizer.tokenize(code)[:block_size-2]
    target_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    target_ids =  tokenizer.convert_tokens_to_ids(target_tokens)
    target_mask = [1] *len(target_ids)
    padding_length = block_size - len(target_ids)
    target_ids+=[tokenizer.pad_token_id]*padding_length
    target_mask+=[0]*padding_length   

    nl = js['factors']
    nl_tokens=tokenizer.tokenize(nl)[:block_size-2]
    source_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask = [1] * (len(source_tokens))
    source_mask+=[0]*padding_length

    return InputFeatures(source_ids,source_mask,target_ids,target_mask)

class CodeData(Dataset):
    def __init__(self, tokenizer, dataset):
        self.examples = []
        for i in tqdm(range(len(dataset)),desc = "Processing dataset..."):
          x = dataset.iloc[i]
          self.examples.append(convert_examples_to_features(x,tokenizer,30))
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, indx):       
        return torch.tensor(self.examples[indx].input_ids),torch.tensor(self.examples[indx].input_mask),torch.tensor(self.examples[indx].target_ids),torch.tensor(self.examples[indx].target_mask)

blocksize = 30
config= RobertaConfig.from_pretrained('microsoft/codebert-base')
tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base',config=config)

class Beam(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence

class Seq2Seq(nn.Module):   
    def __init__(self, config,tokenizer,beam_size, max_length,sos_id,eos_id):
        super(Seq2Seq, self).__init__()
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.tokenizer=tokenizer
        self.encoder = RobertaModel.from_pretrained('microsoft/codebert-base',config=config)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)    
        
    def forward(self,source_ids,source_mask,target_ids=None,target_mask=None): 
        
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        #if target_ids is not None:  
        attn_mask=-1e4 *(1-self.bias[:target_ids.shape[1],:target_ids.shape[1]])
        tgt_embeddings = self.encoder.embeddings(target_ids).permute([1,0,2]).contiguous()
        out = self.decoder(tgt_embeddings,encoder_output,tgt_mask=attn_mask,memory_key_padding_mask=(1-source_mask).bool())
        hidden_states = torch.tanh(self.dense(out)).permute([1,0,2]).contiguous()
        lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
        active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        #print(active_loss,shift_logits.shape,shift_labels.shape)
            # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],shift_labels.view(-1)[active_loss])

        outputs = loss,loss*active_loss.sum(),active_loss.sum()
        return outputs

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)


model = Seq2Seq(tokenizer= tokenizer, config=config,beam_size= 10,max_length=30,sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id).to(device)

train_batch_size = 128
torch.cuda.empty_cache()
print("Training data...")
train_dataset = CodeData(tokenizer, train_data)
print("Validation data...")
val_dataset = CodeData(tokenizer,val_data)

train_dataloader = DataLoader(train_dataset, 
                                  batch_size=train_batch_size,shuffle=True,num_workers=2)
val_dataloader = DataLoader(val_dataset, 
                                  batch_size=train_batch_size,shuffle = False,num_workers=2)
num_train_epochs= 5

optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()],
         }]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr= 0.00005)
    
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
  torch.save(model.state_dict(), 'CodeBERTmodel-nlp-{}.pkl'.format(idx+1))

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.plot(train_loss_graph,'k')
plt.plot(val_loss_graph,'y')
plt.legend(["Training Loss","Validation Loss"])
plt.savefig('Plot-Loss-PY-CodeBERT.png')
