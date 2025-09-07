from unittest import TestCase, mock
from unittest.mock import MagicMock

from mlworkflows.etl.tokenizer import TokenizerFactory


class TestTokenizerFactory(TestCase):

    @mock.patch('mlworkflows.etl.tokenizer.AutoTokenizer.from_pretrained')
    def test_create_base(self, mock_from_pretrained):
        mock_from_pretrained.return_value = 'tokenizer'
        tokenizer_factory = TokenizerFactory(
            name_or_path='bert-base-uncased',
            use_fast=True,
            cache_dir=None,
            add_tokens_to_vocab=[],
            shared=False
        )
        self.assertEqual(0, mock_from_pretrained.call_count)
        self.assertEqual('tokenizer', tokenizer_factory.create())
        mock_from_pretrained.assert_called_once_with('bert-base-uncased', use_fast=True, cache_dir=None, trust_remote_code=False)

    @mock.patch('mlworkflows.etl.tokenizer.AutoTokenizer.from_pretrained')
    def test_create_shared(self, mock_from_pretrained):
        mock_from_pretrained.return_value = 'tokenizer'
        tokenizer_factory = TokenizerFactory(
            name_or_path='bert-base-uncased',
            use_fast=True,
            cache_dir=None,
            add_tokens_to_vocab=[],
            shared=True
        )
        self.assertEqual(0, mock_from_pretrained.call_count)
        self.assertEqual('tokenizer', tokenizer_factory.create())
        self.assertTrue(tokenizer_factory.shared)
        self.assertEqual(1, mock_from_pretrained.call_count)
        self.assertEqual('tokenizer', tokenizer_factory.create())
        self.assertEqual(1, mock_from_pretrained.call_count)

    @mock.patch('mlworkflows.etl.tokenizer.AutoTokenizer.from_pretrained')
    def test_create_add_tokens_to_vocab(self, mock_from_pretrained):
        mock_from_pretrained.return_value = MagicMock()
        tokenizer_factory = TokenizerFactory(
            name_or_path='bert-base-uncased',
            use_fast=True,
            cache_dir=None,
            add_tokens_to_vocab=['[SPECIAL_TOKEN]'],
            shared=False
        )
        self.assertEqual(0, mock_from_pretrained.call_count)
        tokenizer = tokenizer_factory.create()
        mock_from_pretrained.assert_called_once_with('bert-base-uncased', use_fast=True, cache_dir=None, trust_remote_code=False)
        tokenizer.add_tokens.assert_called_once_with(['[SPECIAL_TOKEN]'])


