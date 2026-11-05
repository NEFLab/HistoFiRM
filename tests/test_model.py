import copy
import numpy as np
import torch
from histofirm.model import HistoFiRM
from histofirm.stream import moments, backward_encoder
from histofirm.prediction import local_readout


def make_model():
    torch.manual_seed(11)
    cfg = dict(
        input_dim=6,
        hidden_dim=8,
        latent_dim=4,
        n_targets=3,
        n_cancers=2,
        adapter_rank=2,
        program_count=2,
        program_hidden=5,
        program_score_bound=6.0,
        aux_weight=0.05,
    )
    m = HistoFiRM(cfg).double()
    m.init_program_axis([0, 1])
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * 0.15)
        m.base_coef.copy_(torch.randn_like(m.base_coef) * 0.2)
        m.base_bias.copy_(torch.randn_like(m.base_bias))
        m.y_scale.fill_(1.7)
        m.y_center.fill_(0.4)
    return m


def test_streamed_predictions_and_gradients_match_full_bag():
    m = make_model()
    full = copy.deepcopy(m)
    x = torch.randn(29, 6, dtype=torch.float64)
    y = torch.tensor([[0.2, -0.3, 0.1]], dtype=torch.float64)
    mask = torch.ones_like(y, dtype=torch.bool)
    w = torch.ones(1)
    z = full.encode(x)
    p = full.forward_bag(x, 0)[None]
    e = full.extra_features(z).mean(0, keepdim=True)
    loss = (p - y).square().mean() + 0.05 * full.auxiliary_losses(e, y, mask, w)
    loss.backward()
    chunks = lambda: [x[i : i + 7].numpy() for i in range(0, len(x), 7)]
    st = moments(m, chunks, "cpu", 0)
    hm = st["hmean"][None]
    mm = st["mean"][None].requires_grad_(True)
    ss = st["second"][None].requires_grad_(True)
    ee = st["extra"][None].requires_grad_(True)
    sp = m.forward_stats(hm, mm, ss, torch.tensor([0]), ee)
    sl = (sp - y).square().mean() + 0.05 * m.auxiliary_losses(ee, y, mask, w)
    sl.backward()
    backward_encoder(m, chunks, len(x), mm.grad[0], ss.grad[0], "cpu", ee.grad[0], 0)
    torch.testing.assert_close(sp, p, rtol=1e-11, atol=1e-11)
    for (_, a), (_, b) in zip(m.named_parameters(), full.named_parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-10)


def test_local_fields_reconstruct_patient_and_preserve_na():
    m = make_model()
    x = torch.randn(37, 6, dtype=torch.float64)
    chunks = lambda: [x[i : i + 9].numpy() for i in range(0, len(x), 9)]
    response, heterogeneity = local_readout(m, chunks, 0)
    expected = m.decode(m.forward_bag(x, 0)[None], torch.tensor([0])).detach().numpy()[0]
    np.testing.assert_allclose((response + heterogeneity).mean(0), expected, rtol=1e-10, atol=1e-10)
    with torch.no_grad():
        m.supported[0, 1] = False
        m.variable[0, 2] = False
    r, h = local_readout(m, chunks, 0)
    assert np.isnan(r[:, 1]).all() and np.isnan(h[:, 1]).all()
    np.testing.assert_allclose(r[:, 2], m.y_center[0, 2])
    assert (h[:, 2] == 0).all()


def test_chunk_size_does_not_change_complete_bag_output():
    m = make_model()
    x = torch.randn(31, 6, dtype=torch.float64)
    results = []
    for size in [1, 7, 100]:
        s = moments(m, lambda: [x[i : i + size].numpy() for i in range(0, len(x), size)], "cpu", 0)
        results.append(
            m.forward_stats(
                s["hmean"][None],
                s["mean"][None],
                s["second"][None],
                torch.tensor([0]),
                s["extra"][None],
            )
        )
    for p in results[1:]:
        torch.testing.assert_close(p, results[0], rtol=1e-10, atol=1e-10)
