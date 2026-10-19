import numpy as np

from concept_analysis.projector import LEACEProjection
from word_circles.roseg import fit_leace_chunked


def test_chunked_leace_matches_the_authors_one_shot_fit():
    rng = np.random.default_rng(0)
    H = 16
    # duration (frames) -> (n_tokens, L, H), like the authors' grouped training data
    X_group = {L: rng.standard_normal((n, L, H)).astype(np.float32) + np.linspace(0, 3, L)[None, :, None]
               for L, n in ((5, 40), (7, 30), (9, 25))}

    # The authors' path (estimation.get_temporal_subspace): flatten every group, label = frame index.
    X = np.concatenate([X_group[L].reshape(-1, H) for L in sorted(X_group)])
    y = np.concatenate([np.tile(np.arange(L), len(X_group[L])) for L in sorted(X_group)])
    one_shot = LEACEProjection(name="LEACE_temporal").fit(X, y)

    chunked = fit_leace_chunked(X_group, chunk_frames=37)

    probe = rng.standard_normal((50, H)).astype(np.float32)
    a, b = one_shot.transform(probe, n_components=2), chunked.transform(probe, n_components=2)
    signs = np.sign(np.sum(a * b, axis=0))  # singular vectors are defined up to sign
    np.testing.assert_allclose(b * signs, a, rtol=1e-3, atol=1e-3)
    np.testing.assert_array_equal(chunked.classes_, one_shot.classes_)
