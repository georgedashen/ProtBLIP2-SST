# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Stage-2 data module: Galactica joint [START_AMINO]seq[END_AMINO] Summary Protein: prefix."""
import re

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from data_provider.gal_helpers import escape_custom_split_sequence
from data_provider.stage1_dm import SwissProtDataset, OntoProteinDataset

JOINT_SUMMARY_SUFFIX = " Summary Protein:"


def format_aa_joint_summary_prefix(seq: str) -> str:
    """Galactica paper-style contiguous prefix ending with Summary Protein:"""
    seq = re.sub(r"[UZOB]", "X", seq)
    wrapped = f"[START_AMINO]{seq}[END_AMINO]"
    return escape_custom_split_sequence(wrapped) + JOINT_SUMMARY_SUFFIX


class Stage2AArawJointCollater(object):
    def __init__(self, tokenizer, text_max_len, prot_max_len):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len

    def __call__(self, batch):
        prot_seqs, _prompt_seqs, text_seqs, indices = zip(*batch)
        prefixes = [format_aa_joint_summary_prefix(seq) for seq in prot_seqs]
        full_texts = [pre + txt for pre, txt in zip(prefixes, text_seqs)]
        self.tokenizer.padding_side = "right"
        enc = self.tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.prot_max_len + self.text_max_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        prefix_lens = []
        for pre in prefixes:
            prefix_lens.append(
                len(self.tokenizer(pre, add_special_tokens=True).input_ids)
            )
        return enc, prefix_lens


class InferenceAArawJointCollater(object):
    def __init__(self, tokenizer, text_max_len, prot_max_len):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.prot_max_len = prot_max_len

    def __call__(self, batch):
        prot_seqs, _prompt_seqs, text_seqs, indices = zip(*batch)
        prefixes = [format_aa_joint_summary_prefix(seq) for seq in prot_seqs]
        self.tokenizer.padding_side = "right"
        prefix_batch = self.tokenizer(
            prefixes,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            max_length=self.prot_max_len,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        target_dict = {"targets": text_seqs, "indices": indices}
        return prefix_batch, target_dict


class Stage2AArawJointDM(LightningDataModule):
    def __init__(self, root: str = "data/", args=None):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = args.num_workers
        self.text_max_len = args.text_max_len
        self.prot_max_len = args.prot_max_len

        suffix = getattr(args, "suffix", "")
        test = getattr(args, "test", 0)
        if root.find("SwissProt") >= 0:
            self.train_dataset = SwissProtDataset(
                root + f"/train_set{suffix}.json",
                prompt="Swiss-Prot description: ",
                return_prompt=True,
                test=test,
            )
            self.val_dataset = SwissProtDataset(
                root + f"/valid_set{suffix}.json",
                prompt="Swiss-Prot description: ",
                return_prompt=True,
                test=test,
            )
            self.test_dataset = SwissProtDataset(
                root + f"/test_set{suffix}.json",
                prompt="Swiss-Prot description: ",
                return_prompt=True,
                test=0,
            )
        elif root.find("OntoProteinDatasetV2") >= 0:
            self.train_dataset = OntoProteinDataset(
                root + "/train.txt", prompt="Gene Ontology description: ", return_prompt=True
            )
            self.val_dataset = OntoProteinDataset(
                root + "/valid.txt", prompt="Gene Ontology description: ", return_prompt=True
            )
            self.test_dataset = OntoProteinDataset(
                root + "/test.txt", prompt="Gene Ontology description: ", return_prompt=True
            )
        else:
            raise NotImplementedError

        self.tokenizer = None

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _stage2_collater(self):
        return Stage2AArawJointCollater(self.tokenizer, self.text_max_len, self.prot_max_len)

    def _inference_collater(self):
        return InferenceAArawJointCollater(self.tokenizer, self.text_max_len, self.prot_max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=False,
            collate_fn=self._stage2_collater(),
        )

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=self._stage2_collater(),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=False,
            collate_fn=self._inference_collater(),
        )
        return [val_loader, test_loader]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module (AA joint Summary Protein)")
        parser.add_argument("--num_workers", type=int, default=2)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--inference_batch_size", type=int, default=4)
        parser.add_argument("--root", type=str, default="data/SwissProtV3")
        parser.add_argument("--text_max_len", type=int, default=128)
        parser.add_argument("--prot_max_len", type=int, default=1024)
        return parent_parser
