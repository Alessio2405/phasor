"""Residue summaries: causality, exactness, and train/decode agreement.

If the training-time summaries and the decoding-time summaries disagree, the
model trains on a different function from the one it generates with, and no
amount of good perplexity means anything. That is what
:func:`test_training_matches_incremental_decoding` guards.
"""

import pytest
import torch

from phasor.summaries import (
    IncrementalSummaries,
    causal_bucket_summaries,
    gather_summaries_for_queries,
    summary_block_index,
)


def test_summary_block_index_never_reaches_the_future():
    for stride in (1, 2, 4, 8):
        idx = summary_block_index(64, stride)
        for t in range(64):
            blk = int(idx[t])
            if blk < 0:
                continue
            last_position_included = (blk + 1) * stride - 1
            assert last_position_included <= t, (
                f"stride={stride} t={t}: summary includes position "
                f"{last_position_included} which is in the future"
            )


def test_stride_one_is_the_exact_running_mean():
    Bsz, H, T, D, B = 2, 3, 17, 5, 6
    x = torch.randn(Bsz, H, T, D)
    bucket = torch.randint(0, B, (Bsz, H, T))

    means, counts = causal_bucket_summaries(x, bucket, B, stride=1)
    m, c = gather_summaries_for_queries(means, counts, 1, T)

    for t in (0, 1, T // 2, T - 1):
        for b in range(B):
            sel = bucket[0, 0, : t + 1] == b
            if sel.sum() == 0:
                assert c[0, 0, t, b] == 0
                continue
            expected = x[0, 0, : t + 1][sel].mean(dim=0)
            torch.testing.assert_close(m[0, 0, t, b], expected, atol=1e-5, rtol=1e-4)
            assert int(c[0, 0, t, b]) == int(sel.sum())


def test_future_keys_cannot_change_past_summaries():
    """Direct perturbation test: change the last token, earlier summaries hold."""
    Bsz, H, T, D, B = 1, 1, 24, 4, 6
    x = torch.randn(Bsz, H, T, D)
    bucket = torch.randint(0, B, (Bsz, H, T))

    m1, c1 = gather_summaries_for_queries(*causal_bucket_summaries(x, bucket, B), 1, T)

    x2 = x.clone()
    x2[0, 0, T - 1] += 100.0  # violent perturbation of the final key
    m2, c2 = gather_summaries_for_queries(*causal_bucket_summaries(x2, bucket, B), 1, T)

    torch.testing.assert_close(m1[:, :, : T - 1], m2[:, :, : T - 1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(c1, c2)


def test_summaries_are_causal_by_autograd():
    """The general version: no gradient path from a future key to a past summary."""
    Bsz, H, T, D, B = 1, 1, 16, 4, 6
    x = torch.randn(Bsz, H, T, D, requires_grad=True)
    bucket = torch.randint(0, B, (Bsz, H, T))

    means, counts = causal_bucket_summaries(x, bucket, B, stride=1)
    m, _ = gather_summaries_for_queries(means, counts, 1, T)

    t = 5
    m[0, 0, t].sum().backward()
    assert x.grad is not None
    future_grad = x.grad[0, 0, t + 1 :].abs().sum()
    assert future_grad.item() == 0.0, "summary at t depends on keys after t"


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_training_matches_incremental_decoding(stride):
    """Batched cumsum path == token-by-token state, at the same read points."""
    T, D, B = 32, 6, 8
    k = torch.randn(1, 1, T, D)
    v = torch.randn(1, 1, T, D)
    bucket = torch.randint(0, B, (1, 1, T))

    mk, ck = causal_bucket_summaries(k, bucket, B, stride)
    mv, _ = causal_bucket_summaries(v, bucket, B, stride)
    kbar, cnt = gather_summaries_for_queries(mk, ck, stride, T)
    vbar, _ = gather_summaries_for_queries(mv, ck, stride, T)

    state = IncrementalSummaries(B, D, D)
    idx = summary_block_index(T, stride)

    for t in range(T):
        # Feed everything the block boundary for t would have absorbed.
        end = (int(idx[t]) + 1) * stride - 1
        while state.counts.sum() < end + 1:
            j = int(state.counts.sum())
            state.update(int(bucket[0, 0, j]), k[0, 0, j], v[0, 0, j])

        k_ref, v_ref, c_ref = state.read()
        valid = c_ref > 0
        torch.testing.assert_close(cnt[0, 0, t][valid], c_ref[valid], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(kbar[0, 0, t][valid], k_ref[valid], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(vbar[0, 0, t][valid], v_ref[valid], atol=1e-5, rtol=1e-4)


def test_empty_buckets_report_zero_count():
    """Callers must be able to mask never-seen buckets out of the softmax."""
    T, D, B = 8, 3, 10
    bucket = torch.zeros(1, 1, T, dtype=torch.long)  # everything in bucket 0
    means, counts = causal_bucket_summaries(torch.randn(1, 1, T, D), bucket, B)
    _, c = gather_summaries_for_queries(means, counts, 1, T)
    assert (c[0, 0, :, 1:] == 0).all()
    assert (c[0, 0, :, 0] > 0).all()


def test_padding_does_not_leak_into_counts():
    """Sequence lengths that are not multiples of the stride must still be exact."""
    T, D, B, stride = 10, 3, 4, 4  # 10 is not a multiple of 4
    x = torch.randn(1, 1, T, D)
    bucket = torch.randint(0, B, (1, 1, T))
    means, counts = causal_bucket_summaries(x, bucket, B, stride)
    _, c = gather_summaries_for_queries(means, counts, stride, T)
    # At t = T-1 the last complete block ends at index 7, so 8 keys are counted.
    assert int(c[0, 0, T - 1].sum()) == 8


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        causal_bucket_summaries(torch.randn(1, 1, 8, 4), torch.zeros(1, 1, 7, dtype=torch.long), 4)
