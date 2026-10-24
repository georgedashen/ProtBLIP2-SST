from esm.models.esmc import ESMC
from esm.sdk.api import *
import torch
import os
import pickle
from esm.tokenization import EsmSequenceTokenizer
 
# 使用预下载的参数
os.environ["INFRA_PROVIDER"] = "True"
device = torch.device("cuda:0")
client = ESMC.from_pretrained("esmc_600m",device=device)
 
def get_esm_embedding(seq):
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True))
    esm_embedding = logits_output.embeddings
    assert isinstance(esm_embedding,torch.Tensor)
    return protein_tensor, esm_embedding
 
def decode_esm_embedding(protein_tensor):
    protein = client.decode(protein_tensor)
    seq = protein.sequence
    return seq
 
seq = "AAAAAAAAAAACCCCCCCCCCEEEEEEEEEEE"
print(seq)
 
# 获取序列embedding
protein_tensor, emb = get_esm_embedding(seq)
# 这里需要绕过一下assert，所以使用了unsqueeze
protein_tensor.sequence = protein_tensor.sequence.unsqueeze(0)
print(protein_tensor.sequence.shape)
with open("seq_emb.pkl","wb") as f:
    pickle.dump(emb,f)
print(emb.shape)
seq = decode_esm_embedding(protein_tensor)
print(seq)