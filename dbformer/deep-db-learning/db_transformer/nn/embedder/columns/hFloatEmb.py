import torch
import torch.nn as nn

class SimpleRepeater(nn.Module):

    def __init__(self, dim=256) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("scaler", torch.logspace(-2, 1, dim))
        self.register_buffer("floatscaler", torch.logspace(-2.5, 1, dim))
        self.floatenc = getfloatenc(dim)

    def forward(self, feat: torch.Tensor):
        assert feat.ndim == 1
        if torch.is_floating_point(feat):


            return self.floatenc(feat.unsqueeze(-1))
        else:
            feat = feat.unsqueeze(1).to(torch.float) * self.scaler
            return torch.cos(feat)

def getfloatenc(hiddim: int=256, train: bool=False):

    middim = int(hiddim**0.5+0.1)
    FloatEnc = nn.Sequential(nn.Linear(1, middim), nn.LayerNorm(middim, elementwise_affine=False), nn.SiLU(inplace=True), nn.Linear(middim, middim), nn.LayerNorm(middim, elementwise_affine=False), nn.SiLU(inplace=True), nn.Linear(middim, hiddim), nn.LayerNorm(hiddim, elementwise_affine=False), nn.SiLU(inplace=True),)
    if not train:
        FloatEnc.load_state_dict(torch.load(f"floatenc-{hiddim}.pt", map_location="cuda", weights_only=True))
        FloatEnc.eval()
        for p in FloatEnc.parameters():
            p.requires_grad_(False)
        FloatEnc = FloatEnc
    else:
        FloatEnc.train()
    return FloatEnc

def getfloatdec(hiddim: int=256, train: bool=False):

    FloatDec = nn.Sequential(nn.LayerNorm(hiddim, elementwise_affine=False), nn.Linear(hiddim, 1, bias=False))
    if not train:
        FloatDec.load_state_dict(torch.load(f"floatdec-{hiddim}.pt", map_location="cuda", weights_only=True))
        FloatDec.eval()
        for p in FloatDec.parameters():
            p.requires_grad_(False)
    else:
        FloatDec.train()
    return FloatDec

if __name__ == "__main__":
    hiddim = 512
    floatdec = getfloatdec(hiddim, train=True)
    floatenc = getfloatenc(hiddim, train=True)
    for p in floatdec.parameters():
        p.requires_grad_(True)
    for p in floatenc.parameters():
        p.requires_grad_(True)
    device = torch.device("cuda")
    floatdec, floatenc = floatdec.to(device), floatenc.to(device)
    optimizer = torch.optim.AdamW(list(floatdec.parameters())+list(floatenc.parameters()), lr=1e-3, weight_decay=1e-3)
    bestloss = 1000
    for i in range(100000):
        x = torch.randn((65536*16, 1), device=device)
        emb = floatenc(x)
        y = floatdec(emb) - x
        loss = y.square().mean() + 0.1 * emb.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100==0:
            floatdec.eval()
            floatenc.eval()
            with torch.no_grad():
                x = torch.randn((65536*32, 1), device=device)
                y = floatdec(floatenc(x)) - x
                tloss = y.abs().mean().item()
            print(i, tloss, y.abs().max().item(), flush=True)
            if tloss < bestloss:
                print("save")
                torch.save(floatdec.state_dict(), f"floatdec-{hiddim}.pt")
                torch.save(floatenc.state_dict(), f"floatenc-{hiddim}.pt")
                bestloss = tloss
            floatdec.train()
            floatenc.train()