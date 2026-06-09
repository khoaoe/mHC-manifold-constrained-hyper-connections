"""Test for PaperMetricsCollector."""
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../examples/qwen2.5-0.5B/")))
from metrics_collector import PaperMetricsCollector

class DummyConfig:
    def __init__(self):
        self.output_attentions = False

class DummyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = DummyConfig()

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyBaseModel()
        self.hc_blocks = None
        self.collect_metrics = False

def test_collector_flags_and_metrics():
    """Ensure collector toggles flags and computes metrics properly."""
    model = DummyModel()
    collector = PaperMetricsCollector(model, n_streams=4)
    
    # Enable
    collector.enable_collection()
    assert model.collect_metrics is True
    assert model.model.config.output_attentions is True
    
    # Simulate forward pass populating the lists
    model.model._attn_weights = [torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)] * 2
    model._residual_norms = [10.0, 12.0]
    
    # Compute
    metrics = collector.compute_metrics()
    
    assert metrics["avg_attn_entropy"] > 0
    assert metrics["residual_norm_final"] == 12.0
    assert len(metrics["residual_norm_layer_curve"]) == 2
    
    # Disable
    collector.disable_collection()
    assert model.collect_metrics is False
    assert model.model.config.output_attentions is False
    assert len(model._residual_norms) == 0
    assert len(model.model._attn_weights) == 0
