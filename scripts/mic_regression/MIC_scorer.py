import os
import sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from AMPGen.MIC_scorer.scorer import LSTMModel  # model class
from esm import Alphabet, pretrained, MSATransformer

class MIC_scorer:
    def __init__(
        self,
        scaler_data_path="AMPGen/MIC_scorer/Scorer_model/ecoliscaler.pkl",
        model_path="AMPGen/MIC_scorer/Scorer_model/2ecoli_best_model_checkpoint.pth",
        esm_model_location='esm2_t36_3B_UR50D',
        repr_layer=36,
        truncation_seq_length=1022,
        to_device='cuda'
    ):
        # store config
        self.repr_layer = repr_layer
        self.truncation_seq_length = truncation_seq_length
        self.to_device = 'cuda' if torch.cuda.is_available() and to_device == 'cuda' else 'cpu'

        # Load scaler once
        with open(scaler_data_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Load ESM model and alphabet (safe fallback to model name if needed)
        try:
            self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(esm_model_location)
        except Exception:
            model_name = os.path.basename(str(esm_model_location)).split('.')[0]
            self.esm_model, self.alphabet = pretrained.load_model_and_alphabet(model_name)

        if isinstance(self.esm_model, MSATransformer):
            raise ValueError("MSA models not supported for single-sequence embedding")

        self.esm_model.eval()
        if self.to_device == 'cuda':
            self.esm_model = self.esm_model.cuda()

        # Batch converter (use truncation if supported by API)
        try:
            self.batch_converter = self.alphabet.get_batch_converter(self.truncation_seq_length)
        except TypeError:
            self.batch_converter = self.alphabet.get_batch_converter()

        # Load predictor model once
        # instantiate with correct input_size later after seeing scaler dimension
        sample_feature_dim = getattr(self.scaler, "n_features_in_", None)
        if sample_feature_dim is None:
            # fallback: infer from scaler.mean_ if available
            sample_feature_dim = len(getattr(self.scaler, "mean_", []))
        self.model = LSTMModel(input_size=sample_feature_dim, hidden_size=128, num_layers=2, output_size=1, dropout_rate=0.7)
        state = torch.load(model_path, map_location='cpu')
        if isinstance(state, dict) and 'model_state_dict' in state:
            self.model.load_state_dict(state['model_state_dict'])
        else:
            # if file is a pure state_dict, load directly
            try:
                self.model.load_state_dict(state)
            except RuntimeError:
                # try single-level dict with keys like 'state_dict'
                if 'state_dict' in state:
                    self.model.load_state_dict(state['state_dict'])
                else:
                    raise
        self.model.to(self.to_device)
        self.model.eval()

        # Optional warm-up small batch to avoid retracing / cold starts
        try:
            dummy = torch.zeros((1, 1, sample_feature_dim), dtype=torch.float32, device=self.to_device)
            with torch.no_grad():
                _ = self.model(dummy)
        except Exception:
            pass

    # ---------------------------------------------------------------------
    #                 PREDICT FUNCTION — RUNS EVERY TIME
    # ---------------------------------------------------------------------
    def predict(self, seq_list, batch_size=16):
        """
        seq_list: list[str]
        returns: list[float] predicted values (same order)
        """
        if not seq_list:
            return []

        # Compute ESM mean embeddings in manageable batches
        mean_reps = []
        for start in range(0, len(seq_list), batch_size):
            chunk = seq_list[start:start + batch_size]
            batch_tuples = [(str(i + start + 1), s) for i, s in enumerate(chunk)]
            labels, strs, toks = self.batch_converter(batch_tuples)
            toks = toks.to(device=self.to_device if self.to_device == 'cuda' else 'cpu')
            with torch.no_grad():
                out = self.esm_model(toks, repr_layers=[self.repr_layer], return_contacts=False)
            reps = out["representations"][self.repr_layer].to(device='cpu').numpy()
            for i, seq in enumerate(strs):
                seq_len = min(len(seq), self.truncation_seq_length)
                rep = reps[i, 1: seq_len + 1].mean(0)  # numpy array
                mean_reps.append(rep)

        X_emb = np.array(mean_reps, dtype=np.float32)
        # scale
        X_scaled = self.scaler.transform(X_emb)

        # create torch loader and predict
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.to_device)
        dataset = TensorDataset(X_tensor.unsqueeze(1))  # shape (N, 1, features)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in loader:
                xb = batch[0]
                out = self.model(xb)
                preds.extend(out.squeeze(-1).cpu().numpy().tolist())

        return preds


# predictor = MIC_scorer()

# test_peptides = [
#     {"sequence":"GIGKFLHSAGKFGKAFVGEIMNS", "log10_MIC": 1.204},
#     {"sequence":"LLGDFFRKSKEKIGKEFKRIVQRIKDFLR", "log10_MIC": 0.60},
#     {"sequence":"ILPWKWPWWPWRR", "log10_MIC": 0.75},
#     {"sequence":"RGGRLCYCRRRFCVCVGR", "log10_MIC": -0.301},
#     {"sequence":"FVQWFSKFLGRIL", "log10_MIC": 1.000},
#     {"sequence":"KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQAAT", "log10_MIC": 0.15},
#     {"sequence":"GIINTLQKYYCRVRGGRCAVLSCLPKEEQIGKCSTRGRKCCRRKK", "log10_MIC": -0.15},
#     {"sequence":"GIGAVLKVLTTGLPALISWIKRKRQQ", "log10_MIC": 0.15},
#     {"sequence":"TRSSRAGLQFPVGRVHRLLRK", "log10_MIC": 0.30},
#     {"sequence":"KRIVQRIKDFLR", "log10_MIC": 0.90},
#     {"sequence":"KWCYIYKQGRCY", "log10_MIC": 0.00},
#     {"sequence":"FLPLIGRVLSGIL", "log10_MIC": 1.398},
#     {"sequence":"FFHHIFRGIVHVGKTIHRL", "log10_MIC": 0.00},
#     {"sequence":"GKLNLKGLKGLLK", "log10_MIC": 0.60},
#     {"sequence":"GWGTVGKLFKGSVR", "log10_MIC": 0.30},
#     {"sequence":"GSKKPVPIIYCNRRTGKCQRM", "log10_MIC": -0.15},
#     {"sequence":"KKLLKKLLKKLL", "log10_MIC": 0.70},
#     {"sequence":"RRWFRR", "log10_MIC": 0.90},
#     {"sequence":"LFLFLFLFLF", "log10_MIC": 1.30},
#     {"sequence":"GIKHLKAGLAK", "log10_MIC": 0.00}
# ]

# # Convert to list of sequences
# seq_list = [x["sequence"] for x in test_peptides]
# expected_list = [x["log10_MIC"] for x in test_peptides]

# # -------------------------
# # Call your existing MIC function
# # -------------------------

# preds = predictor.predict(
#     seq_list,
# )

# # -------------------------
# # Print comparison table
# # -------------------------

# print("\n=== MIC Prediction Comparison ===")
# print("{:<35} {:>12} {:>14} {:>14}".format(
#     "Sequence", "Expected", "Predicted", "Abs Error"
# ))

# for seq, exp, pred in zip(seq_list, expected_list, preds):
#     err = abs(float(pred) - exp)
#     print("{:<35} {:>12.3f} {:>14.3f} {:>14.3f}".format(
#         seq[:33] + ("…" if len(seq) > 33 else ""),
#         exp,
#         pred,
#         err
#     ))

# # -------------------------
# # Optional summary
# # -------------------------
# mae = float(np.mean([abs(float(p) - e) for p, e in zip(preds, expected_list)]))
# print("\nMean Absolute Error (MAE): {:.3f}".format(mae))


