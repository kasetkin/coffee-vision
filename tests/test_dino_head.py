"""Tests for coffeecv_dino.head, the depth-0 logistic-regression head. Plain unittest.

Run from the repo root:  python -m unittest discover -s tests -p "test_dino*" -v

On real data: DINOv3 features of real bean patches from the cam_iphone crops (coffeecv_dino.reference),
12 per class across its 8 classes, split alternately into fit and validation halves. Skips when the
weights or the crops are not on this machine.
"""
import unittest

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from coffeecv_dino.backbone import MODELS_PRETRAINED, SPECS, build_backbone
from coffeecv_dino.head import export_linear, fit_head, predict
from coffeecv_dino.reference import have_reference_data, reference_patches

HAVE = (MODELS_PRETRAINED / SPECS["dinov3_vits16"].weights).exists() and have_reference_data()


@unittest.skipUnless(HAVE, "DINOv3 weights or the cam_iphone crops are missing")
class TestHeadOnRealFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y, cls.class_ids, cls.class_labels = reference_patches(per_class=12)
        bb = build_backbone("dinov3_vits16")
        with torch.no_grad():
            feats = torch.cat([bb.features(x[i:i + 32])["cls_mean"] for i in range(0, len(x), 32)]).numpy()
        fit_rows = np.arange(len(y)) % 2 == 0
        cls.Xtr, cls.ytr = feats[fit_rows], y[fit_rows]
        cls.Xva, cls.yva = feats[~fit_rows], y[~fit_rows]

    def test_folded_scaler_reproduces_sklearn(self):
        """One nn.Linear must give the logits of StandardScaler -> LogisticRegression."""
        scaler = StandardScaler().fit(self.Xtr)
        clf = LogisticRegression(C=0.5, max_iter=3000).fit(scaler.transform(self.Xtr), self.ytr)
        with torch.no_grad():
            ours = export_linear(clf, scaler)(torch.from_numpy(self.Xva)).numpy()
        np.testing.assert_allclose(ours, clf.decision_function(scaler.transform(self.Xva)),
                                   atol=1e-3, rtol=1e-5)

    def test_selects_on_val_and_classifies(self):
        fit = fit_head(self.Xtr, self.ytr, self.Xva, self.yva, self.class_ids, self.class_labels)
        best = max(fit.val_macro_f1_by_C.values())
        # The chosen C scores best on val, and ties go to the smallest such C (strongest regularisation).
        self.assertEqual(fit.C, min(c for c, s in fit.val_macro_f1_by_C.items() if s == best))
        pred, probs, _ = predict(fit.linear, self.Xva)
        self.assertEqual(probs.shape, (len(self.yva), len(self.class_ids)))
        self.assertGreater((pred == self.yva).mean(), 0.5, "far above the 1/8 chance level expected")

    def test_refuses_a_class_missing_from_training(self):
        keep = self.ytr != self.ytr.max()
        with self.assertRaisesRegex(ValueError, "expected 0.."):
            fit_head(self.Xtr[keep], self.ytr[keep], self.Xva, self.yva, self.class_ids, self.class_labels)


if __name__ == "__main__":
    unittest.main()
