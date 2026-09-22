from sot.config import CommitConfig, MoTConfig, SelectorConfig
from sot.thresholds import apply_threshold_policy


def test_threshold_policy_prefers_dataset_over_global() -> None:
    config = MoTConfig(
        selector=SelectorConfig(gate_prob_threshold=0.5),
        commit=CommitConfig(stop_threshold=0.5),
    )
    policy = {
        "global": {"base_gate_prob_threshold": 0.4, "base_stop_threshold": 0.6},
        "per_dataset": {
            "math": {"best_gate_prob_threshold": 0.3, "best_stop_threshold": 0.7}
        },
    }
    applied = apply_threshold_policy(config, policy, dataset="math")
    assert applied.selector.gate_prob_threshold == 0.3
    assert applied.commit.stop_threshold == 0.7
    assert config.selector.gate_prob_threshold == 0.5


def test_released_qwen_embedding_policy_is_explicit() -> None:
    config = MoTConfig(
        selector=SelectorConfig(gate_prob_threshold=0.3390347313135862),
        commit=CommitConfig(stop_threshold=0.5811020839410002),
    )
    policy = {
        "global": {"base_gate_prob_threshold": 0.475, "base_stop_threshold": 0.5},
        "per_dataset": {},
    }
    applied = apply_threshold_policy(config, policy, dataset="longbench_multifieldqa")
    assert applied.selector.gate_prob_threshold == 0.475
    assert applied.commit.stop_threshold == 0.5
