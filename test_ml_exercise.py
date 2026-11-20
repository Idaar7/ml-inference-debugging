import unittest

import numpy as np

from ml_exercise import predict_positive, split_data, train_model


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train_x, cls.test_x, train_y, _ = split_data()
        cls.model = train_model(train_x, train_y)

    def test_inference_uses_fitted_scaler(self):
        raw_rows = self.test_x[:10]
        scaler = self.model.named_steps["standardscaler"]
        classifier = self.model.named_steps["logisticregression"]
        expected = classifier.predict_proba(scaler.transform(raw_rows))[:, 1]
        np.testing.assert_allclose(predict_positive(self.model, raw_rows), expected)

    def test_single_row_matches_batch(self):
        batch = self.test_x[:10]
        batch_scores = predict_positive(self.model, batch)
        single_score = predict_positive(self.model, batch[4:5])

        self.assertEqual(batch_scores.shape, (10,))
        self.assertEqual(single_score.shape, (1,))
        self.assertTrue(np.isfinite(batch_scores).all())
        self.assertTrue(((batch_scores >= 0) & (batch_scores <= 1)).all())
        self.assertAlmostEqual(single_score[0], batch_scores[4])


if __name__ == "__main__":
    unittest.main()
