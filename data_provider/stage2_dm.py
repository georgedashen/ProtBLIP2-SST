# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset
from data_provider.plm_embed_cache import (
    PlmEmbedCacheStore,
    add_plm_cache_args,
    prepare_plm_embed_caches,
)


class Stage2Collater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len, plm_cache=None):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        self.plm_cache = plm_cache
        
    def __call__(self, batch):
        prot_seqs, prompt_seqs, text_seqs, indices = zip(*batch)
        if False:
            self.tokenizer.padding_side = 'left'
            prompt_tokens = self.tokenizer(prompt_seqs,
                                        truncation=True,
                                        padding='longest',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True, 
                                        return_token_type_ids=False)
            self.tokenizer.padding_side = 'right'
            text_tokens = self.tokenizer(text_seqs,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True, 
                                        return_token_type_ids=False)
        else:
            self.tokenizer.padding_side = 'left'
            prompt_tokens = self.tokenizer(prompt_seqs,
                                           truncation=True,
                                           padding='longest',
                                           add_special_tokens=True,
                                           max_length=self.text_max_len,
                                           return_tensors='pt',
                                           return_attention_mask=True, 
                                           return_token_type_ids=False)
            max_prompt_len = int(prompt_tokens.attention_mask.sum(dim=1).max())
            input_pair = [[p, t] for p, t in zip(prompt_seqs, text_seqs)]
            input_tokens = self.tokenizer(input_pair,
                                          truncation=True,
                                          padding='max_length',
                                          add_special_tokens=True,
                                          max_length=self.text_max_len + max_prompt_len,
                                          return_tensors='pt',
                                          return_attention_mask=True,
                                          return_token_type_ids=True)
        if self.plm_cache is not None:
            # Keep cached embeddings on CPU in DataLoader workers (fork-safe).
            prot_batch = self.plm_cache.get(indices)
            return prot_batch, input_tokens
        prot_tokens = self.prot_tokenizer(prot_seqs,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.prot_max_len,
                                          return_tensors="pt",
                                          return_attention_mask=True, 
                                          return_token_type_ids=False)
        return prot_tokens, input_tokens


class InferenceCollater(object):
    def __init__(self, tokenizer, prot_tokenizer, text_max_len, prot_max_len, plm_cache=None):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        self.plm_cache = plm_cache
        
    def __call__(self, batch):
        prot_seqs, prompt_seqs, text_seqs, indices = zip(*batch)
        self.tokenizer.padding_side = 'right'
        prompt_tokens = self.tokenizer(prompt_seqs,
                                       truncation=True,
                                       padding='longest',
                                       add_special_tokens=True,
                                       max_length=self.text_max_len,
                                       return_tensors='pt',
                                       return_attention_mask=True, 
                                       return_token_type_ids=False)
        if self.plm_cache is not None:
            # Keep cached embeddings on CPU in DataLoader workers (fork-safe).
            prot_batch = self.plm_cache.get(indices)
        else:
            prot_batch = self.prot_tokenizer(prot_seqs,
                                             truncation=True,
                                             padding='max_length',
                                             max_length=self.prot_max_len,
                                             return_tensors="pt",
                                             return_attention_mask=True, 
                                             return_token_type_ids=False)
        target_dict = {'targets': text_seqs, 'indices': indices}
        return prot_batch, prompt_tokens, target_dict


class Stage2DM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len
        # self.prompt = args.prompt
        
        suffix = getattr(args, 'suffix', '')
        test = getattr(args, 'test', 0)
        if root.find('SwissProt') >= 0:
            self.train_dataset = SwissProtDataset(root+f'/train_set{suffix}.json', prompt='Swiss-Prot description: ', return_prompt=True, test=test)
            self.val_dataset = SwissProtDataset(root+f'/valid_set{suffix}.json', prompt='Swiss-Prot description: ', return_prompt=True, test=test)
            self.test_dataset = SwissProtDataset(root+f'/test_set{suffix}.json', prompt='Swiss-Prot description: ', return_prompt=True, test=0)
        elif root.find('OntoProteinDatasetV2') >= 0:
            self.train_dataset = OntoProteinDataset(root+'/train.txt', prompt='Gene Ontology description: ', return_prompt=True)
            self.val_dataset = OntoProteinDataset(root+'/valid.txt', prompt='Gene Ontology description: ', return_prompt=True)
            self.test_dataset = OntoProteinDataset(root+'/test.txt', prompt='Gene Ontology description: ', return_prompt=True)
        else:
            raise NotImplementedError

        self.tokenizer = None
        self.prot_tokenizer = None
        self.plm_cache_store = PlmEmbedCacheStore()
    
    def init_tokenizer(self, tokenizer, prot_tokenizer):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer

    def _stage2_collater(self, dataset):
        cache = self.plm_cache_store.for_dataset(dataset)
        return Stage2Collater(
            self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, plm_cache=cache
        )

    def _inference_collater(self, dataset):
        cache = self.plm_cache_store.for_dataset(dataset)
        return InferenceCollater(
            self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, plm_cache=cache
        )

    def _cache_dataset_pairs(self):
        suffix = getattr(self.args, 'suffix', '')
        if self.args.root.find('SwissProt') >= 0:
            return [
                (self.train_dataset, self.args.root + f'/train_set{suffix}.json'),
                (self.val_dataset, self.args.root + f'/valid_set{suffix}.json'),
                (self.test_dataset, self.args.root + f'/test_set{suffix}.json'),
            ]
        return [
            (self.train_dataset, self.train_dataset.data_path),
            (self.val_dataset, self.val_dataset.data_path),
            (self.test_dataset, self.test_dataset.data_path),
        ]

    def prepare_plm_embed_cache(self, plm_module, is_saprot: bool = False):
        prepare_plm_embed_caches(
            self.plm_cache_store,
            self._cache_dataset_pairs(),
            plm_module,
            self.prot_tokenizer,
            self.args,
            is_saprot=is_saprot,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=self._stage2_collater(self.train_dataset),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=self._stage2_collater(self.val_dataset),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=self._inference_collater(self.test_dataset),
        )
        return [val_loader, test_loader]
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--q_max_len', type=int, default=29)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prompt', type=str, default='The protein has the following properties: ')
        parser.add_argument('--filter_side_qa', action='store_true', default=False)
        add_plm_cache_args(parser)
        return parent_parser



class Stage2MixDM(LightningDataModule):
    def __init__(
        self,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len
        # self.prompt = args.prompt
        assert args.mix_dataset
        
        train_dataset1 = SwissProtDataset(root+'/SwissProtV3/train_set.json', prompt='Swiss-Prot description: ', return_prompt=True)
        train_dataset2 = OntoProteinDataset(root+'/OntoProteinDatasetV2/train.txt', prompt='Gene Ontology description: ', return_prompt=True)
        self.train_dataset = ConcatDataset([train_dataset1, train_dataset2])
        self.swiss_val_dataset = SwissProtDataset(root+'/SwissProtV3/valid_set.json', prompt='Swiss-Prot description: ', return_prompt=True)
        self.onto_val_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/valid.txt', prompt='Gene Ontology description: ', return_prompt=True)
        self.swiss_test_dataset = SwissProtDataset(root+'/SwissProtV3/test_set.json', prompt='Swiss-Prot description: ', return_prompt=True)
        self.onto_test_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/test.txt', prompt='Gene Ontology description: ', return_prompt=True)
        
        self.tokenizer = None
        self.prot_tokenizer = None
        self.plm_cache_store = PlmEmbedCacheStore()
    
    def init_tokenizer(self, tokenizer, prot_tokenizer):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer

    def _stage2_collater(self, dataset):
        cache = self.plm_cache_store.for_dataset(dataset)
        return Stage2Collater(
            self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, plm_cache=cache
        )

    def _inference_collater(self, dataset):
        cache = self.plm_cache_store.for_dataset(dataset)
        return InferenceCollater(
            self.tokenizer, self.prot_tokenizer, self.text_max_len, self.prot_max_len, plm_cache=cache
        )

    def _cache_dataset_pairs(self):
        root = self.args.root
        return [
            (self.train_dataset, "mix_train_concat"),
            (self.swiss_val_dataset, self.swiss_val_dataset.data_path),
            (self.onto_val_dataset, self.onto_val_dataset.data_path),
            (self.swiss_test_dataset, self.swiss_test_dataset.data_path),
            (self.onto_test_dataset, self.onto_test_dataset.data_path),
        ]

    def prepare_plm_embed_cache(self, plm_module, is_saprot: bool = False):
        prepare_plm_embed_caches(
            self.plm_cache_store,
            self._cache_dataset_pairs(),
            plm_module,
            self.prot_tokenizer,
            self.args,
            is_saprot=is_saprot,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=self._stage2_collater(self.train_dataset),
        )
        return loader

    def val_dataloader(self):
        swiss_val_loader = DataLoader(
            self.swiss_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=self._stage2_collater(self.swiss_val_dataset),
        )
        swiss_test_loader = DataLoader(
            self.swiss_test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=self._inference_collater(self.swiss_test_dataset),
        )

        onto_val_loader = DataLoader(
            self.onto_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=self._stage2_collater(self.onto_val_dataset),
        )
        onto_test_loader = DataLoader(
            self.onto_test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=self._inference_collater(self.onto_test_dataset),
        )
        return [swiss_val_loader, swiss_test_loader, onto_val_loader, onto_test_loader]
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--q_max_len', type=int, default=29)
        parser.add_argument('--a_max_len', type=int, default=36)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        add_plm_cache_args(parser)
        return parent_parser


