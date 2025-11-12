import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, num_classes: int, proj_dim: int = 512, dropout_p: float = 0.0):
        super().__init__()
        self.proj_dim = proj_dim

        self.proj_l = None
        self.proj_b = None
        self.proj_r = None

        self.norm = nn.LayerNorm(proj_dim * 3)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(proj_dim * 3, num_classes)

    @staticmethod
    def _maybe_pool(x):
        return x.mean(dim=1) if x.dim() == 3 else x

    def _init_projs(self, Dl, Db, Dr, device, dtype):
        # create the three projection layers with known in_features
        self.proj_l = nn.Linear(Dl, self.proj_dim, bias=False, device=device, dtype=dtype)
        self.proj_b = nn.Linear(Db, self.proj_dim, bias=False, device=device, dtype=dtype)
        self.proj_r = nn.Linear(Dr, self.proj_dim, bias=False, device=device, dtype=dtype)

    def forward(self, input_l, input_b, input_r):
        # 1) pool
        l = self._maybe_pool(input_l)
        b = self._maybe_pool(input_b)
        r = self._maybe_pool(input_r)

        # 2) FORCE FP32 for stability (and to avoid Half/Float mismatch)
        l = l.to(dtype=torch.float32)
        b = b.to(dtype=torch.float32)
        r = r.to(dtype=torch.float32)

        # 3) init projection layers with correct in_features on first pass
        if self.proj_l is None:
            device = l.device
            dtype  = torch.float32  # keep everything in fp32
            self._init_projs(l.size(-1), b.size(-1), r.size(-1), device, dtype)

        # 4) projections
        lp = self.proj_l(l)  # [B, proj_dim]
        bp = self.proj_b(b)
        rp = self.proj_r(r)

        # 5) concat + norm/dropout
        x = torch.cat([lp, bp, rp], dim=1)
        x = self.norm(x)
        x = self.dropout(x)

        # 6) final safety cast to classifier dtype
        x = x.to(dtype=self.classifier.weight.dtype)
        return self.classifier(x)
