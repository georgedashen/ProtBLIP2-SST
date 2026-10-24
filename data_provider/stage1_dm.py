# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import json
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import random, re
from pathlib import Path

from data_provider.plm_embed_cache import (
    PlmEmbedCacheStore,
    add_plm_cache_args,
    prepare_plm_embed_caches,
    preprocess_prot_seqs,
)


def rand_seq_crop(seq, max_len):
    if len(seq) <= max_len:
        return seq
    rand_pos = random.randint(0, len(seq)-1-max_len)
    return seq[rand_pos:rand_pos+max_len]


class Stage1Collater(object):
    def __init__(
        self,
        tokenizer,
        prot_tokenizer,
        text_max_len,
        prot_max_len,
        prot_aug='None',
        plm_name="Rostlab/prot_t5_xl_uniref50",
        plm_cache=None,
    ):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len
        self.prot_aug = prot_aug
        self.plm_name = plm_name
        self.plm_cache = plm_cache
    
    def __call__(self, batch):
        prot_seqs, text_seqs, indices = zip(*batch)
        if self.prot_aug == 'rand_crop':
            prot_seqs = [rand_seq_crop(seq, self.prot_max_len-2) for seq in prot_seqs] # -2 for the two special tokens
        
        text_tokens = self.tokenizer(text_seqs,
                                     truncation=True,
                                     padding='max_length',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True, 
                                     return_token_type_ids=False)
        if self.plm_cache is not None:
            # Keep cached embeddings on CPU in DataLoader workers (fork-safe).
            prot_batch = self.plm_cache.get(indices)
            return prot_batch, text_tokens

        prot_seqs = preprocess_prot_seqs(prot_seqs, self.plm_name)
        prot_tokens = self.prot_tokenizer(prot_seqs,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.prot_max_len,
                                          return_tensors="pt",
                                          return_attention_mask=True, 
                                          return_token_type_ids=False)
        return prot_tokens, text_tokens


class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len
        if root.find('SwissProt') >= 0:
            self.train_dataset = SwissProtDataset(root+f'/train_set{args.suffix}.json', test=args.test)
            self.val_dataset = SwissProtDataset(root+f'/valid_set{args.suffix}.json', test=args.test)
            self.test_dataset = SwissProtDataset(root+f'/test_set{args.suffix}.json', test=0)
            self.val_dataset_match = SwissProtDataset(root+f'/valid_set{args.suffix}.json', test=args.test).shuffle()
            self.test_dataset_match = SwissProtDataset(root+f'/test_set{args.suffix}.json', test=0).shuffle()
        elif root.find('PDBDataset') >= 0:
            self.train_dataset = PDBAbstractDataset(root, 'train.txt')
            self.val_dataset = PDBAbstractDataset(root, 'val.txt')
            self.test_dataset = PDBAbstractDataset(root, 'test.txt')
            self.val_dataset_match = PDBAbstractDataset(root, 'val.txt').shuffle()
            self.test_dataset_match = PDBAbstractDataset(root, 'test.txt').shuffle()    
        elif root.find('OntoProtein') >= 0:
            self.train_dataset = OntoProteinDataset(root+'/train.txt')
            self.val_dataset = OntoProteinDataset(root+'/valid.txt')
            self.test_dataset = OntoProteinDataset(root+'/test.txt')
            self.val_dataset_match = OntoProteinDataset(root+'/valid.txt').shuffle()
            self.test_dataset_match = OntoProteinDataset(root+'/test.txt').shuffle()
        else:
            raise NotImplementedError

        self.tokenizer = None
        self.prot_tokenizer = None
        self.prot_aug = args.prot_aug
        self.plm_name = args.plm_name
        self.args = args
        self.plm_cache_store = PlmEmbedCacheStore()
        self._is_saprot = False

    def init_tokenizer(self, tokenizer, prot_tokenizer):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer

    def _stage1_collater(self, dataset):
        cache = self.plm_cache_store.for_dataset(dataset)
        return Stage1Collater(
            self.tokenizer,
            self.prot_tokenizer,
            self.text_max_len,
            self.prot_max_len,
            self.prot_aug,
            self.plm_name,
            plm_cache=cache,
        )

    def _cache_dataset_pairs(self):
        return [
            (self.train_dataset, self.train_dataset.data_path),
            (self.val_dataset, self.val_dataset.data_path),
            (self.test_dataset, self.test_dataset.data_path),
            (self.val_dataset_match, f"{self.val_dataset.data_path}:match"),
            (self.test_dataset_match, f"{self.test_dataset.data_path}:match"),
        ]

    def prepare_plm_embed_cache(self, plm_module, is_saprot: bool = False):
        self._is_saprot = is_saprot
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
            # persistent_workers=True,
            collate_fn=self._stage1_collater(self.train_dataset)
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            # persistent_workers=True,
            collate_fn=self._stage1_collater(self.val_dataset)
        )
        return loader

    def match_dataloader(self):
        val_match_loader = DataLoader(self.val_dataset_match, 
                                      batch_size=self.match_batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers, 
                                      pin_memory=False, 
                                      drop_last=False, 
                                    #   persistent_workers=True,
                                      collate_fn=self._stage1_collater(self.val_dataset_match))
        test_match_loader = DataLoader(self.test_dataset_match, 
                                       batch_size=self.match_batch_size,
                                       shuffle=False,
                                       num_workers=self.num_workers, 
                                       pin_memory=False, 
                                       drop_last=False, 
                                    #    persistent_workers=True,
                                       collate_fn=self._stage1_collater(self.test_dataset_match))
        return val_match_loader, test_match_loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prot_aug', type=str, default='None')
        add_plm_cache_args(parser)
        return parent_parser
    


class Stage1MixDM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len
        assert args.mix_dataset
        train_dataset1 = SwissProtDataset(root+'/SwissProtV3/train_set.json')
        train_dataset2 = OntoProteinDataset(root+'/OntoProteinDatasetV2/train.txt')
        self.train_dataset = ConcatDataset([train_dataset1, train_dataset2], )
        self.swiss_val_dataset = SwissProtDataset(root+'/SwissProtV3/valid_set.json')
        self.onto_val_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/valid.txt')
        self.swiss_test_dataset = SwissProtDataset(root+'/SwissProtV3/test_set.json')
        self.onto_test_dataset = OntoProteinDataset(root+'/OntoProteinDatasetV2/test.txt')
        
        self.swiss_val_dataset_match = SwissProtDataset(root+'/SwissProtV3/valid_set.json').shuffle()
        self.onto_val_dataset_match = OntoProteinDataset(root+'/OntoProteinDatasetV2/valid.txt').shuffle()
        self.swiss_test_dataset_match = SwissProtDataset(root+'/SwissProtV3/test_set.json').shuffle()
        self.onto_test_dataset_match = OntoProteinDataset(root+'/OntoProteinDatasetV2/test.txt').shuffle()
        
        self.tokenizer = None
        self.prot_tokenizer = None
        self.prot_aug = args.prot_aug
        self.plm_name = args.plm_name
        self.args = args
        self.plm_cache_store = PlmEmbedCacheStore()

    def init_tokenizer(self, tokenizer, prot_tokenizer):
        self.tokenizer = tokenizer
        self.prot_tokenizer = prot_tokenizer

    def _stage1_collater(self, dataset):
        cache = self.plm_cache_store.for_dataset(dataset)
        return Stage1Collater(
            self.tokenizer,
            self.prot_tokenizer,
            self.text_max_len,
            self.prot_max_len,
            self.prot_aug,
            self.plm_name,
            plm_cache=cache,
        )

    def _cache_dataset_pairs(self):
        return [
            (self.train_dataset, "mix_train_concat"),
            (self.swiss_val_dataset, self.swiss_val_dataset.data_path),
            (self.onto_val_dataset, self.onto_val_dataset.data_path),
            (self.swiss_test_dataset, self.swiss_test_dataset.data_path),
            (self.onto_test_dataset, self.onto_test_dataset.data_path),
            (self.swiss_val_dataset_match, f"{self.swiss_val_dataset.data_path}:match"),
            (self.onto_val_dataset_match, f"{self.onto_val_dataset.data_path}:match"),
            (self.swiss_test_dataset_match, f"{self.swiss_test_dataset.data_path}:match"),
            (self.onto_test_dataset_match, f"{self.onto_test_dataset.data_path}:match"),
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
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True,
            collate_fn=self._stage1_collater(self.train_dataset)
        )
        return loader

    def val_dataloader(self):
        loader1 = DataLoader(
            self.swiss_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=False,
            collate_fn=self._stage1_collater(self.swiss_val_dataset)
        )
        loader2 = DataLoader(
            self.onto_val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=False,
            collate_fn=self._stage1_collater(self.onto_val_dataset)
        )
        return [loader1, loader2]

    def swiss_match_dataloader(self):
        val_match_loader = DataLoader(self.swiss_val_dataset_match,
                                      batch_size=self.match_batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      prefetch_factor=4,
                                      drop_last=False,
                                      collate_fn=self._stage1_collater(self.swiss_val_dataset_match))
        test_match_loader = DataLoader(self.swiss_test_dataset_match, 
                                       batch_size=self.match_batch_size,
                                       shuffle=False,
                                       num_workers=self.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True,
                                       prefetch_factor=4,
                                       drop_last=False,
                                       collate_fn=self._stage1_collater(self.swiss_test_dataset_match))
        return val_match_loader, test_match_loader
    
    def onto_match_dataloader(self):
        val_match_loader = DataLoader(self.onto_val_dataset_match,
                                      batch_size=self.match_batch_size,
                                      shuffle=False,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      persistent_workers=True,
                                      prefetch_factor=4,
                                      drop_last=False,
                                      collate_fn=self._stage1_collater(self.onto_val_dataset_match))
        test_match_loader = DataLoader(self.onto_test_dataset_match, 
                                       batch_size=self.match_batch_size,
                                       shuffle=False,
                                       num_workers=self.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True,
                                       prefetch_factor=4,
                                       drop_last=False,
                                       collate_fn=self._stage1_collater(self.onto_test_dataset_match))
        return val_match_loader, test_match_loader

    
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--root', type=str, default='data/SwissProtV3')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prot_max_len', type=int, default=1024)
        parser.add_argument('--prot_aug', type=str, default='None')
        add_plm_cache_args(parser)
        return parent_parser


class SwissProtDataset(Dataset):
    def __init__(self, data_path, prompt='Swiss-Prot description: ', return_prompt=False, test=0):
        super(SwissProtDataset, self).__init__()
        self.data_path = data_path

        ## load data
        with open(data_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            self.data_list = [json.loads(line) for line in lines]

        ## preprocessing
        self.data_list = [(p, t.strip() + '\n') for p, t in self.data_list]
        if test > 0:
            self.data_list = self.data_list[:test]

        self.text2id = {}
        for prot_seq, text_seq in self.data_list:
            if text_seq not in self.text2id:
                self.text2id[text_seq] = len(self.text2id)
        
        self.prompt = prompt
        self.return_prompt = return_prompt

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def len(self,):
        return len(self)
    
    def get(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq, text_seq = self.data_list[index]
        if self.return_prompt:
            return prot_seq, self.prompt, text_seq, index
        return prot_seq, text_seq, index


class PDBAbstractDataset(Dataset):
    def __init__(self, root_path, subset, prompt='ABSTRACT: ', return_prompt=False):
        super(PDBAbstractDataset, self).__init__()
        self.data_path = Path(root_path) / subset
        self.abstract_path = Path(root_path) / 'abstract.json'
        
        ## load dataset
        with open(self.abstract_path, 'r') as f:
            abstract_data = json.load(f)
            abstract_data_dict = {line['pdb_id']: line['caption'] for line in abstract_data}
        
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            pdb2seq = [line.strip().split('\t') for line in lines]
        
        ## process dataset
        data_list = []
        for pdb_id, seq in pdb2seq:
            abstract = abstract_data_dict[pdb_id]
            abstract = abstract.replace('\n', ' ').strip() + '\n'
            data_list.append((seq, abstract))
        self.data_list = data_list
        self.prompt = prompt
        self.return_prompt = return_prompt

    def shuffle(self):
        random.shuffle(self.data_list)
        return self
        
    def len(self,):
        return len(self)
    
    def get(self, idx):
        return self.__getitem__(idx)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        seq, abstract = self.data_list[index]
        if self.return_prompt:
            return seq, self.prompt, abstract, index
        return seq, abstract, index
    

class OntoProteinDataset(Dataset):
    def __init__(self, data_path, prompt='Gene Ontology description: ', return_prompt=False):
        super(OntoProteinDataset, self).__init__()
        self.data_path = data_path
        
        ## load data
        with open(data_path, 'r') as f:
            lines = f.readlines()
            self.data_list = [line.strip().split('\t') for line in lines]

        ## preprocessing
        ## fixme: I have disabled the signal word for this dataset. However, it was used in previous experiments.
        if True:
            self.data_list = [(p, t.strip() + '\n') for p, t in self.data_list]
        else:
            self.data_list = [(p, "KG: " + t.strip() + '\n') for p, t in self.data_list]
        self.prompt = prompt
        self.return_prompt = return_prompt

    def shuffle(self):
        random.shuffle(self.data_list)
        return self

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        prot_seq, text_seq = self.data_list[index]
        if self.return_prompt:
            return prot_seq, self.prompt, text_seq, index
        return prot_seq, text_seq, index


if __name__ == '__main__':
    import numpy as np
    ## get statistics for swiss prot dataset
    if True:
        swiss_train = SwissProtDataset('../data/SwissProtV3/train_set.json')
        swiss_valid = SwissProtDataset('../data/SwissProtV3/valid_set.json')
        swiss_test = SwissProtDataset('../data/SwissProtV3/test_set.json')
        print(len(swiss_train), len(swiss_valid), len(swiss_test))
        
        ## get amino acid statistics
        aa_lens = np.asarray([len(seq) for seq, _ in swiss_train.data_list])
        print('Train dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())
        aa_lens = np.asarray([len(seq) for seq, _ in swiss_valid.data_list])
        print('Valid dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())
        aa_lens = np.asarray([len(seq) for seq, _ in swiss_test.data_list])
        print('Test dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())

        ## get text statistics
        text_lens = np.asarray([len(seq.split()) for _, seq in swiss_train.data_list])
        print('Train dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
        text_lens = np.asarray([len(seq.split()) for _, seq in swiss_valid.data_list])
        print('Valid dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
        text_lens = np.asarray([len(seq.split()) for _, seq in swiss_test.data_list])
        print('Test dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
        print('---------------------------')

    ## get statistics for onto protein dataset
    onto_train = OntoProteinDataset('../data/OntoProteinDatasetV2/train.txt')
    onto_valid = OntoProteinDataset('../data/OntoProteinDatasetV2/valid.txt')
    onto_test = OntoProteinDataset('../data/OntoProteinDatasetV2/test.txt')
    print(len(onto_train), len(onto_valid), len(onto_test))

    ## get amino acid statistics
    aa_lens = np.asarray([len(seq) for seq, _ in onto_train.data_list])
    print('Train dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())
    aa_lens = np.asarray([len(seq) for seq, _ in onto_valid.data_list])
    print('Valid dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())
    aa_lens = np.asarray([len(seq) for seq, _ in onto_test.data_list])
    print('Test dataset mean: ', np.mean(aa_lens), 'min: ', aa_lens.min(), 'max: ', aa_lens.max())

    ## get text statistics
    text_lens = np.asarray([len(seq.split()) for _, seq in onto_train.data_list])
    print('Train dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
    text_lens = np.asarray([len(seq.split()) for _, seq in onto_valid.data_list])
    print('Valid dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
    text_lens = np.asarray([len(seq.split()) for _, seq in onto_test.data_list])
    print('Test dataset mean: ', np.mean(text_lens), 'min: ', text_lens.min(), 'max: ', text_lens.max())
    print('---------------------------')
