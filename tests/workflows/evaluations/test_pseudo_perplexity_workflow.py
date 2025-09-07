import os
import shutil
import sys
from io import StringIO
from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from classconfig import DelayedFactory
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizerFast

from mlworkflows.etl.template import Template
from mlworkflows.workflows.evaluations.pseudo_perplexity import PseudoPerplexityWorkflow

SCRIPT_DIR = Path(__file__).parent
TMP_DIR = SCRIPT_DIR / "tmp"


class TestPseudoPerplexityWorkflow(TestCase):
    def setUp(self):
        self.cleanup()
        TMP_DIR.mkdir(parents=True, exist_ok=True)

        self.dataset_mock = MagicMock()
        self.dataset_mock.create.return_value = Dataset.from_dict({
            'text': ["11 2 3", "3 2 11"],
        })
        self.dataset_mock.num_proc = 0

        self.model_fct_mock = MagicMock()
        self.model = MagicMock()

        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        def return_correct_device(input_ids):
            m = MagicMock()
            m.logits = torch.tensor([
                [
                    [100.0, 100.0, 0.0, 0.0],
                    [0.0, 0.0, 100.0, 0.0],
                    [0.0, 0.0, 0.0, 100.0]
                ],
                [
                    [0.0, 100.0, 100.0, 0.0],
                    [0.0, 0.0, 100.0, 0.0],
                    [0.0, 0.0, 0.0, 100.0]
                ],
                [
                    [0.0, 100.0, 0.0, 0.0],
                    [0.0, 0.0, 100.0, 0.0],
                    [0.0, 0.0, 100.0, 100.0]
                ],
                [
                    [0.0, 0.0, 100.0, 100.0],
                    [0.0, 0.0, 100.0, 0.0],
                    [0.0, 1000.0, 0.0, 0.0]
                ],
                [
                    [0.0, 0.0, 0.0, 100.0],
                    [0.0, 0.0, 100.0, 0.0],
                    [0.0, 1000.0, 0.0, 0.0]
                ],
                [
                    [0.0, 0.0, 0.0, 100.0],
                    [0.0, 0.0, 100.0, 0.0],
                    [0.0, 1000.0, 100.0, 0.0]
                ],
            ], device=input_ids.device)
            return m

        self.model.side_effect = return_correct_device

        self.model_fct_mock.create.return_value = self.model

        self.tokenizer_fct_mock = MagicMock()
        self.tokenizer_fct_mock.create.return_value = self.create_tokenizer()

        self.workflow = PseudoPerplexityWorkflow(
            dataset_fct=self.dataset_mock,
            split='train',
            model_fct=self.model_fct_mock,
            tokenizer_fct=self.tokenizer_fct_mock,
            input=DelayedFactory(
                Template,
                args={
                    "template": "{{text}}"
                }
            ),
            pseudo_perplexity_per_sample=TMP_DIR / "pseudo_perplexity_per_sample.jsonl",
            ignore_index=-100,
            attention_mask=False
        )

    @staticmethod
    def cleanup():
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)

    def create_tokenizer(self):
        vocab_dict = {
            "0": 0,
            "11": 1,
            "2": 2,
            "3": 3,
            "[MASK]": 4,
            "[PAD]": 5
        }
        # Create a WordLevel tokenizer from dict
        wordlevel = WordLevel(vocab=vocab_dict, unk_token="[UNK]")

        # Build tokenizers.Tokenizer object
        tokenizer_backend = Tokenizer(wordlevel)
        tokenizer_backend.pre_tokenizer = Whitespace()

        # Wrap in a Hugging Face tokenizer*8888888-
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            pad_token="[PAD]",
            mask_token="[MASK]",
            max_len=100
        )

        return tokenizer

    def test_batch_create_input_text(self):
        # Test the input text creation
        input_text = self.workflow.batch_create_input_text({"text": ["This is a test sentence."],
                                                            "other_field": ["value"]})
        self.assertSequenceEqual(["This is a test sentence."], input_text)

    def test_process_to_ppl_input(self):
        batch = self.workflow.process_to_ppl_input(
            {"text": ["11 2 3"]},
            [42]
        )
        self.assertSequenceEqual([
            [4, 2, 3],
            [1, 4, 3],
            [1, 2, 4]
        ], batch["input_ids"])

        self.assertSequenceEqual([
            [1, -100, -100],
            [-100, 2, -100],
            [-100, -100, 3]
        ], batch["labels"])

        self.assertSequenceEqual([42, 42, 42], batch["sample_id"])

    def test___call__(self):
        backup_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            self.workflow()
            sys.stdout.seek(0)
            output = sys.stdout.read()
            lines = output.splitlines()
            self.assertEquals(len(lines), 12)

            first = CrossEntropyLoss(ignore_index=-100, reduction='none')(
                torch.tensor([
                        [100.0, 100.0, 0.0, 0.0],
                        [0.0, 0.0, 100.0, 0.0],
                        [0.0, 0.0, 0.0, 100.0],

                        [0.0, 100.0, 100.0, 0.0],
                        [0.0, 0.0, 100.0, 0.0],
                        [0.0, 0.0, 0.0, 100.0],

                        [0.0, 100.0, 0.0, 0.0],
                        [0.0, 0.0, 100.0, 0.0],
                        [0.0, 0.0, 100.0, 100.0]
                ]),
                torch.tensor([
                    1, -100, -100,
                    -100, 2, -100,
                    -100, -100, 3
                ])
            )
            second = CrossEntropyLoss(ignore_index=-100, reduction='none')(
                torch.tensor([
                        [0.0, 0.0, 100.0, 100.0],
                        [0.0, 0.0, 100.0, 0.0],
                        [0.0, 1000.0, 0.0, 0.0],

                        [0.0, 0.0, 0.0, 100.0],
                        [0.0, 0.0, 100.0, 0.0],
                        [0.0, 1000.0, 0.0, 0.0],

                        [0.0, 0.0, 0.0, 100.0],
                        [0.0, 0.0, 100.0, 0.0],
                        [0.0, 1000.0, 100.0, 0.0]
                ]),
                torch.tensor([
                    3, -100, -100,
                    -100, 2, -100,
                    -100, -100, 1
                ])
            )
            ppl = torch.hstack([torch.exp(first.sum()/3), torch.exp(second.sum()/3)])

            # PPL
            self.assertAlmostEqual(float(lines[0].split()[-1]), ppl.mean().item())  # AVG
            self.assertAlmostEqual(float(lines[1].split()[-1]), ppl.min().item())  # MIN
            self.assertAlmostEqual(float(lines[2].split()[-1]), ppl.max().item())  # MAX

            char_first = first * torch.tensor([2, 1, 1, 2, 1, 1, 2, 1, 1])
            char_second = second * torch.tensor([1, 1, 2, 1, 1, 2, 1, 1, 2])
            char_ppl = torch.hstack([torch.exp(char_first.sum()/4), torch.exp(char_second.sum()/4)])

            # Char PPL
            self.assertAlmostEqual(float(lines[3].split()[-1]), char_ppl.mean().item())  # AVG
            self.assertAlmostEqual(float(lines[4].split()[-1]), char_ppl.min().item())  # MIN
            self.assertAlmostEqual(float(lines[5].split()[-1]), char_ppl.max().item())  # MAX

            # bits per char
            bits_per_chat = torch.hstack([char_first.sum()/4/torch.log(torch.tensor(2.0)),
                                            char_second.sum()/4/torch.log(torch.tensor(2.0))])
            self.assertAlmostEqual(float(lines[6].split()[-1]), bits_per_chat.mean().item())  # AVG
            self.assertAlmostEqual(float(lines[7].split()[-1]), bits_per_chat.min().item())  # MIN
            self.assertAlmostEqual(float(lines[8].split()[-1]), bits_per_chat.max().item())  # MAX

            # bits per byte same as bits per char here
            self.assertAlmostEqual(float(lines[9].split()[-1]), bits_per_chat.mean().item())  # AVG
            self.assertAlmostEqual(float(lines[10].split()[-1]), bits_per_chat.min().item())  # MIN
            self.assertAlmostEqual(float(lines[11].split()[-1]), bits_per_chat.max().item())  # MAX

        finally:
            sys.stdout = backup_stdout
