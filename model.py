#as implemented by Microsoft using CodeBERT model at https://github.com/microsoft/CodeBERT


import torch
import torch.nn as nn
import beam


class NLPTransformer(nn.Module):   
    """
    Parameters required:
    config = RobertaConfig downloaded from huggingface
    tokenizer = RobertaTokenizer downloaded from huggingface
    beam_size = default 10 as instructed by Microsoft
    max_length = 30 as instructed by Scale AI
    sos_id = start of sequence ID for RobertaTokenizer
    eos_id = end of sequence ID for RobertaTokenizer

    """

    def __init__(self, config,tokenizer,beam_size, max_length,sos_id,eos_id):
        super(NLPTransformer, self).__init__()
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
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)    
        
    def forward(self,source_ids,source_mask,target_ids=None,target_mask=None): 
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        if target_ids is not None:  
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

        else: 
          #if target not given predict target
          preds=[]       
          zero=torch.cuda.LongTensor(1).fill_(0)     
          for i in range(source_ids.shape[0]):
            context=encoder_output[:,i:i+1]
            context_mask=source_mask[i:i+1,:]
            beam = Beam(self.beam_size,self.sos_id,self.eos_id)
            input_ids=beam.getCurrentState()
            context=context.repeat(1, self.beam_size,1)
            context_mask=context_mask.repeat(self.beam_size,1)
            for _ in range(self.max_length): 
              if beam.done():
                break
              attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
              tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
              out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
              out = torch.tanh(self.dense(out))
              hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
              out = self.lsm(self.lm_head(hidden_states)).data
              beam.advance(out)
              input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
              input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
              hyp= beam.getHyp(beam.getFinal())
            pred=beam.buildTargetTokens(hyp)[:self.beam_size]
            pred=[torch.cat([x.view(-1) for x in p]+[zero]*(self.max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))
                
            preds=torch.cat(preds,0)  
            #print(preds[0])              
            return preds   
