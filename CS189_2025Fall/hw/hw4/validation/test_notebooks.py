"""Behavioral checks for the implementations inside all three HW4 notebooks."""
from pathlib import Path
from contextlib import nullcontext
import ast
import json
import random
import re
import types

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def definitions(filename):
    nb = json.loads((ROOT / filename).read_text(encoding='utf-8'))
    ns = dict(np=np, torch=torch, nn=nn, F=F, Dataset=Dataset, DataLoader=DataLoader,
              TensorDataset=TensorDataset, random=random, plt=plt, re=re,
              transforms=torchvision.transforms, torchvision=torchvision, torchaudio=torchaudio,
              Path=Path, nullcontext=nullcontext, PAD_TOKEN='<pad>', START_TOKEN='<start>',
              END_TOKEN='<end>', device=torch.device('cpu'))
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if any(line.lstrip().startswith(('%', '!')) for line in source.splitlines()):
            continue
        tree = ast.parse(source)
        selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), ns)
    return ns


@pytest.fixture(params=['hw4_part1.ipynb', 'hw4_part1_fixed.ipynb'])
def part1(request):
    return definitions(request.param)


def test_no_unfilled_code():
    for path in ROOT.glob('*.ipynb'):
        nb = json.loads(path.read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                if any(line.lstrip().startswith(('%', '!')) for line in source.splitlines()):
                    continue
                assert not any(isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign))
                               and isinstance(node.value, ast.Constant) and node.value.value is Ellipsis
                               for node in ast.walk(ast.parse(source))), path.name


def test_cnn_dimensions_and_gradient(part1):
    model = part1['CNN'](10)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 10)
    nn.CrossEntropyLoss()(logits, torch.tensor([0, 3])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_starter_shape_checks(part1):
    nb = json.loads((ROOT / 'hw4_part1_fixed.ipynb').read_text(encoding='utf-8'))
    part1['SEED'] = 42
    for index in (11, 36, 37, 42, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99):
        exec(''.join(nb['cells'][index]['source']), part1)


def test_residual_identity_and_projection(part1):
    block = part1['ResidualBlock'](8, 8).eval()
    nn.init.zeros_(block.conv2.weight)
    x = torch.rand(2, 8, 13, 13)
    torch.testing.assert_close(block(x), x)
    block = part1['ResidualBlock'](8, 16, initial_downsample=True)
    assert block(x).shape == (2, 16, 7, 7)
    assert part1['ResNet18'](5).eval()(torch.randn(1, 3, 224, 224)).shape == (1, 5)


def test_dataset_grayscale_and_transforms(part1):
    data = [{'image': Image.fromarray(np.full((7, 9), 100, dtype=np.uint8)), 'label': 2}]
    cls = part1['MiniImageNetDataset']
    x, y = cls(data, {2: 'test'})[0]
    assert x.shape == (3, 7, 9) and x.dtype == torch.float32 and y.dtype == torch.long
    x, _ = cls(data, {2: 'test'}, transform=torchvision.transforms.ToTensor())[0]
    torch.testing.assert_close(x, torch.full_like(x, 100/255))


def test_softmax_and_gradients(part1):
    x = torch.tensor([[1000., 1001., -1000.]], requires_grad=True)
    y = part1['softmax'](x)
    torch.testing.assert_close(y, torch.softmax(x, -1))
    assert torch.autograd.gradcheck(part1['softmax'], (torch.randn(2, 3, dtype=torch.double, requires_grad=True),))


def test_attention_mask_and_cross_shapes(part1):
    torch.manual_seed(42)
    q, k, v = torch.randn(2, 3, 4), torch.randn(2, 5, 4), torch.randn(2, 5, 6)
    mask = torch.zeros(3, 5, dtype=torch.bool)
    mask[:, 3:] = True
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
    torch.testing.assert_close(part1['scaled_dot_product_attention'](q, k, v, mask), expected)
    out = part1['scaled_dot_product_attention'](q, k, v, torch.ones_like(mask))
    assert torch.equal(out, torch.zeros_like(out))
    head = part1['MultiHeadAttention'](2, 8)
    assert head(torch.randn(2, 3, 8), torch.randn(2, 5, 8)).shape == (2, 3, 8)


def test_decoder_causality_and_registration(part1):
    model = part1['Transformer'](vocab_size=20, d_model=16, num_heads=2,
                                 num_encoder_layers=1, num_decoder_layers=2).eval()
    x = torch.tensor([[1, 2, 3, 4, 5]])
    changed = x.clone(); changed[:, 3:] = torch.tensor([7, 8])
    torch.testing.assert_close(model(x)[:, :3], model(changed)[:, :3])
    model(x).sum().backward()
    assert model.decoder.layers[0].self_attention.heads[0].W_q.weight.grad is not None
    model.decoder_only = False
    assert model(x, target=x[:, :2]).shape == (1, 2, 20)


def test_positions_and_sequence_targets(part1):
    pe = part1['PositionalEncoding'](7)
    x = pe(torch.zeros(2, 4, 7))
    torch.testing.assert_close(x[0, 0], torch.tensor([0., 1., 0., 1., 0., 1., 0.]))
    assert 'pos' in dict(pe.named_buffers())
    vocab = {'a': 0, 'b': 1, '<pad>': 2, '<start>': 3, '<end>': 4}
    inputs, targets = part1['create_sequences'](['a b'], vocab, max_seq_length=4)
    assert inputs.tolist() == [[3, 0, 1, 2]]
    assert targets.tolist() == [[0, 1, 4, 2]]
    assert part1['create_sequences']([''], vocab, 4)[0].shape == (0, 4)


def test_training_loop_updates_and_optional_validation(part1):
    model = nn.Linear(3, 2)
    before = model.weight.detach().clone()
    loader = DataLoader(TensorDataset(torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 0])), batch_size=3)
    metrics = part1['train'](model, torch.optim.SGD(model.parameters(), lr=0.1), nn.CrossEntropyLoss(), 1, loader)
    assert len(metrics) == 4 and len(metrics[0]) == 1
    assert not torch.equal(before, model.weight)


def test_dna_kmers_and_dataset():
    ns = definitions('hw4_part2.ipynb')
    assert ns['sequence_to_kmer']('ATGCGT', 3) == 'ATG TGC GCG CGT'
    assert ns['sequence_to_kmer']('AT', 6) == ''
    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs['padding'] == 'max_length' and kwargs['max_length'] == 512
            return {'input_ids': torch.ones(1, 512, dtype=torch.long),
                    'attention_mask': torch.ones(1, 512, dtype=torch.long)}
    sample = ns['DNADataset'](['AAA'], [2], Tokenizer())[0]
    assert sample['input_ids'].shape == (512,) and sample['target'].item() == 2
    assert 'target' not in ns['DNADataset'](['AAA'], tokenizer=Tokenizer(), return_targets=False)[0]


def test_audio_dataset_and_normalization(tmp_path):
    import soundfile as sf
    ns = definitions('hw4_part2.ipynb')
    sf.write(tmp_path / '1-3-0-0.wav', np.zeros((1000, 2)), 16000)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)),
                                                torchvision.transforms.Lambda(ns['normalize_spectrogram'])])
    dataset = ns['SpectrogramDataset'](tmp_path, transforms=transform)
    spec, target = dataset[0]
    assert spec.shape == (3, 224, 224) and torch.isfinite(spec).all() and target.item() == 3
    assert torch.equal(dataset[0][0], spec)


def test_convnext_head_and_microbatch_updates():
    ns = definitions('hw4_part2.ipynb')
    model = nn.Module()
    model.classifier = nn.Sequential(nn.LayerNorm(8), nn.Flatten(1), nn.Linear(8, 1000))
    assert ns['replace_final_convnext_linear_layer'](model).classifier[-1].out_features == 10
    a, b = nn.Linear(3, 2), nn.Linear(3, 2)
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(7, 3), torch.tensor([0, 1, 1, 0, 1, 0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=7)
    opt_a, opt_b = torch.optim.SGD(a.parameters(), lr=0.1), torch.optim.SGD(b.parameters(), lr=0.1)
    ns['_train_classifier'](a, opt_a, nn.CrossEntropyLoss(), 'cpu', 1, loader, loader)
    nn.CrossEntropyLoss()(b(x), y).backward(); opt_b.step()
    torch.testing.assert_close(a.weight, b.weight)
    torch.testing.assert_close(a.bias, b.bias)


def test_spectrogram_decibels_preserve_relative_power():
    normalize = definitions('hw4_part2.ipynb')['normalize_spectrogram']
    power = torch.tensor([1., 1e-2, 1e-4, 1e-8]).reshape(1, 1, 4).repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    # Undo ImageNet normalization to check the physical 20 dB intervals.
    actual = normalize(power) * std + mean
    expected = torch.tensor([1., 0.75, 0.5, 0.]).reshape(1, 1, 4).repeat(3, 1, 1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(normalize(torch.zeros_like(power))).all()
    # An overall gain changes absolute power, but not its relative dB pattern.
    torch.testing.assert_close(normalize(power * 10), normalize(power), atol=1e-6, rtol=1e-6)
