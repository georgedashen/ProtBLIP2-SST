# Title: ProtBLIP2-SST: Protein Function Prediction via BLIP2 with Sequence, Structure, and Text

# Abstract
Protein function prediction is a crucial challenge in computational biology that requires information from various aspects including sequence, structure, and text annotation. While multi-modal protein language models (pLM) exist, they are mainly used as embeddings for supervised learning in downsteam tasks given structured numerical or categorical labels, which lack the flexibility of making novel predictions or performing less structured tasks. Recent efforts on utilizing large language models (LLM) to capture protein semantics are in sequence-only or sequence-text alignment styles but do not integrate structure information. To address these limitations, we introduce ProtBLIP2-SST, a model for protein function prediction in text generation manner utilizing both protein sequence and structure. ProtBLIP2-SST uses an protein encoder to integrate sequence and structure, and a BERT-based model as text encoder. To empower LLM to better capture protein semantics, we align protein feature with text feature through contrastive learning as in BLIP2 framework. After the alignment, the protein embeddings from BLIP2 are further concatenated with the prompt embeddings of the LLM for protein captioning fine-tuning. Our results show ProtBLIP2-SST outperforms current models on the SwissProt dataset, with ablation tests demonstrated the effectiveness of integrating structure with sequence information for improved protein understanding.

(reference abstract: 
Bridging protein \textbf{S}equence, \textbf{S}tructure, and \textbf{T}ext (SST) remains difficult for end-to-end generation.

Two gaps: (1) pLMs have a great understanding on sequences or structures but do not on text, while text can help capture fine-grained information which is hard to be encoded (some tokenize text like ESMC, ProTrek). Few integrate sequence, structure, and text together. That is what we want. (2) LLMs are good at many text related task, even including math, but still lack understanding on science, where basic contexts are not natural language but only representations by tokens (DNA, amino acid, chemicals, materials).

We present \textbf{ProtBLIP2-SST}, a BLIP-2-style framework that keeps a frozen protein encoder, an 8-query Q-Former, and a Galactica-1.3B decoder with mid-layer LoRA fixed, while swapping only the encoder: ESM-2 (sequence-only), SaProt (interleaved amino-acid and Foldseek 3Di tokens), or ProstT5 (bilingual sequence--structure T5).

Stage~1 aligns protein and text via contrastive, matching, and captioning losses; stage~2 generates Swiss-Prot captions.
On an aligned 9{,}239-sample test split, SaProt-650M reaches 33.07\% exact match and 58.53 BLEU-4, modestly above ESM-2-650M (32.83\% / 58.12) with complementary errors: structure conditioning reduces fold-specific enzyme confusions, whereas sequence-only encoding remains stronger on short family-label captions.
)
# Introduction

## The task of protein function prediction.

Many deep learning methods have been proposed. GO settings. We treat it as captioning task instead of multi-label multi-class classification. And we also include similarity homology tag and location tag to further evaluate the understanding of protein beyond sequence and function.

Great value and utility of LLM. Recently scientists focused on using LLMs to interpret biological data. But they cannot directly read proteins, since proteins are not text, althought they can be represented as text, eg. amino acid sequences. However, not every aspect of protein can be represented as text, such as protein structure and interaction patterns.

Figure 1 of table showing research gap, models with alignment, modal, LLM-combined.

## Search gap

multimodal-foundation models exist. multi-modal alignment models exist. Sequence-Text aligned models exist. However, sequence-structure-text aligned LLM integrated models do not exist. We try to integrate the three modals. Intuitions are two-fold: current pLM + prediction head + downstream do not understand the protein well, especially build the relationship between local sequence motif and function, and between local structure conformation and function. Or even built by supervisers task labeled specific learning, the relationship and understanding is task specific and implicit. We want to bridge this gap by first, introducing function annotation as text (both function, similarity/homology, and location) and do alignment with the two embeddings via contrastive learning, so that the annotation is encoded into sequence and structure, and can be further utilized for various downstream tasks. More importantly, since we map sequence-structure embedding with text embedding, it will be intrinsically easier to integrate protein information into natural language based methods or tools, especially LLM. Then we can use these embeddings to finetune LLM and then be able to interact with LLM to perform supervised, few-shot, or even zero-shot learning.



## Our method is as following

collect structures from alphafold database v4 and filter swiss-prot database records only retain those with structure available (one contribution, a sequence, structure, and function annotation dataset). Then we encode sequence and structure together using SaProt (one contribution, previous work like ProtT3 use sequence encoder only, and other work design several attention or alignment module for different modal data, which could make training expensive and difficult). We adapt the same structure as ProtT3 using Qformer in BLIP2, the 8-query-token setting, the three-task-losses, and the BIOMEDICAL BERT text encoder. We use flash attetntion as well to accelerate the training. Results in stage 1 show our three-modal structure-integrated model ProtBLIP2-SST performs better than ProtT3. Then in stage 2 we use the 8-query-token + proj head + 'Swiss-Prot description: ' prompt to finetune Galactica, a scientific LLM which trained with protein data. We fintuned Galactica-1.3B with the swiss-prot annotation used in stage1 (formatted output) and evaluate it. Results show our model outperforms both ProtT3 and Prot2Text-V2. Ablation study show better than zero-shot from Galactica, LoRA finetuning Galactica, finetuning with unaligned Esm2 embeddings, and finetuning with ProtT3. Case study show clear evidence of improving structure understanding of our model against others.

# Methods

## Models

Figure 2 of model architecture Figure 1 and stage1/2 training overview. description as above. Model description (Protein encoder, text encoder, Qformer, cross attention). Necessary formula with notations. Two stage training. Stage1 alignment and Stage2 LLM finetune protein captioning.

## Dataset

following the setting in ProtT3. collect afdb data and filter.

## experiment settings

batch size, match batch size, learning rate, epochs, models used in two stages, optimizor.

## Evaluation Metrics

following the setting in ProtT3. Accuracy, recall@20 rerank with 128 candidate in Stage1. exact match, bleu2/4, rouge1/2/L, and meteor in Stage2.

## Experimental Setup

All experiments are conducted on a computer with an ?type, ?CPU, and ?memory. All training and evaluation experiments are run on two A100 GPU each with 80G memory. text and protein model. hyperparameters LoRA, query token, batch size, matching batch size, learning rate schedule, optimizer, epochs in stage1. llm model, mid lora, learning rate and optimizer, batch size and epoch in stage2. flash and deepspeed enabled for acceleration.

# Results

## Good in Stage 1 protein-text retrieval task

Table 1 on main performance
 - compare to ProtT3, Prot2Text-V2, other text related models (ProtChat), other structure models (ProstT5, ESMC)

Table 2 on ablation test.
 - No PTM, No LM
 - Different query numbers
 - Filtering afdb

Figure 3 showing the help of alignment (contrastive learning), refers to Prot2Text-V2

## Good in Stage 2 protein captioning task

Table 3 on main performance
 - Compare to ProtT3, Prot2Text, other text related models (ProtChat), other structure models (ProstT5, ESMC)
 - Binning test similar to Prot2Text-V2

Table 4 on ablation test.
 - Compare to other prompt like finetuning and zeroshot

Figure 4 showing win counts and case study.


# Conclusion
## limitation and future work
More advanced multi-modal foundation model test in stage 1 and 2.
Prompt design to be more versatile.