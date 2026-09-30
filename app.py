"""
Zero-DCE++ + SE-Block + DenoiseHead — interactive test app.

Run with:
    streamlit run app.py

Put this file in the same folder as your trained checkpoint
(default: best_zerodce_pp_seblock_v9.pth), or point CHECKPOINT_PATH
in the sidebar at wherever it actually lives.
"""

import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms.functional as TF

# ──────────────────────────────────────────────────────────────────
# Model definition — must stay IDENTICAL to the training notebook
# (zero_dce_pp_seblock_v9.ipynb) or state_dict loading will fail.
# ──────────────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    _ACT = {
        'relu':  lambda: nn.ReLU(inplace=True),
        'prelu': lambda: nn.PReLU(),
        'silu':  lambda: nn.SiLU(inplace=True),
        'gelu':  lambda: nn.GELU(),
    }
    _GATE = {
        'sigmoid':     lambda: nn.Sigmoid(),
        'hardsigmoid': lambda: nn.Hardsigmoid(inplace=True),
    }

    def __init__(self, channels, reduction=4, activation='relu', gate='sigmoid'):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            self._ACT[activation](),
            nn.Linear(channels // reduction, channels, bias=False),
            self._GATE[gate](),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class DepthSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                    groups=in_channels, bias=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DCENetLite(nn.Module):
    def __init__(self, filters=32, use_attention=True, se_positions=(3,),
                 reduction=4, se_activation='relu', se_gate='sigmoid', iterations=8):
        super().__init__()
        self.use_attention = use_attention
        self.se_positions  = set(se_positions) if use_attention else set()
        self.iterations    = iterations

        self.conv1 = DepthSeparableConv2d(3,          filters)
        self.conv2 = DepthSeparableConv2d(filters,    filters)
        self.conv3 = DepthSeparableConv2d(filters,    filters)
        self.conv4 = DepthSeparableConv2d(filters,    filters)
        self.conv5 = DepthSeparableConv2d(filters*2,  filters)
        self.conv6 = DepthSeparableConv2d(filters*2,  filters)
        self.conv7 = DepthSeparableConv2d(filters*2,  3)
        self.relu  = nn.ReLU(inplace=True)

        if use_attention:
            self.se_blocks = nn.ModuleDict({
                str(pos): SEBlock(filters, reduction=reduction,
                                   activation=se_activation, gate=se_gate)
                for pos in self.se_positions
            })

    def _se(self, feat, pos):
        if pos in self.se_positions:
            return self.se_blocks[str(pos)](feat)
        return feat

    def forward(self, x):
        x1 = self._se(self.relu(self.conv1(x)),  1)
        x2 = self._se(self.relu(self.conv2(x1)), 2)
        x3 = self._se(self.relu(self.conv3(x2)), 3)
        x4 = self._se(self.relu(self.conv4(x3)), 4)
        x5 = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        a_map = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))  # [B, 3, H, W]
        return a_map


class DenoiseHead(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.conv1 = DepthSeparableConv2d(3, channels)
        self.conv2 = DepthSeparableConv2d(channels, 3)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, enhanced):
        residual = self.relu(self.conv1(enhanced))
        residual = torch.tanh(self.conv2(residual)) * 0.1
        return torch.clamp(enhanced + residual, 0, 1)


def exposure_gate(x, low=0.15, high=0.55):
    """Global strength multiplier for a_map. 1.0 for genuinely dark images
    (mean brightness at/below `low`), smoothly fading to 0.0 as the image's
    own average brightness approaches `high` -- so already-well-lit input
    gets little to no enhancement, instead of the network always assuming
    it's looking at a dark LOL-v2-style crop."""
    mean_brightness = x.mean(dim=[1, 2, 3], keepdim=True)
    return torch.clamp((high - mean_brightness) / (high - low), 0.0, 1.0)


def enhance_image_with_curve(x, a_map, iterations=8, max_exposure=0.95, knee=0.10):
    """Same soft-ceiling curve as the v9 training/eval notebooks."""
    enhanced = x
    for _ in range(iterations):
        enhanced = enhanced + a_map * (enhanced.pow(2) - enhanced)
    enhanced = torch.clamp(enhanced, 0.0, 1.0)

    threshold = max_exposure - knee
    over = (enhanced > threshold).float()
    t = ((enhanced - threshold) / (1.0 - threshold + 1e-8)).clamp(0, 1)
    compressed = threshold + knee * (1.0 - torch.exp(-3.0 * t))
    compressed = torch.minimum(compressed, torch.full_like(enhanced, max_exposure))

    return enhanced * (1 - over) + compressed * over


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def load_model(checkpoint_path, se_positions, reduction, se_activation, se_gate, use_denoiser):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = DCENetLite(use_attention=True, se_positions=se_positions, reduction=reduction,
                        se_activation=se_activation, se_gate=se_gate).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    denoise_head = None
    if use_denoiser:
        denoise_head = DenoiseHead().to(device)
        denoise_head.load_state_dict(ckpt['denoise_head'])
        denoise_head.eval()

    return model, denoise_head, device


def run_inference(image: Image.Image, model, denoise_head, device, max_exposure, knee,
                  use_gate=True, gate_low=0.15, gate_high=0.55):
    inp = TF.to_tensor(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        a_map = model(inp)
        
        gate_val = 1.0
        if use_gate:
            gate_tensor = exposure_gate(inp, low=gate_low, high=gate_high)
            gate_val = gate_tensor.item()
            a_map = a_map * gate_tensor

        enhanced = enhance_image_with_curve(inp, a_map, max_exposure=max_exposure, knee=knee)
        if denoise_head is not None:
            enhanced = denoise_head(enhanced)
            
    enhanced_np = enhanced.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return (enhanced_np * 255).astype(np.uint8), gate_val


# ──────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Zero-DCE++ Tester", layout="wide")
st.title("Zero-DCE++ with SE-Block + DenoiseHead — Test Your Own Photo")

with st.sidebar:
    st.header("Model config")
    st.caption("Must match whatever your checkpoint was actually trained with.")
    checkpoint_path = st.text_input("Checkpoint path", value="best_zerodce_pp_seblock_v9.pth")
    se_positions_str = st.text_input("SE positions (comma-separated)", value="3")
    se_positions = tuple(int(p.strip()) for p in se_positions_str.split(",") if p.strip())
    reduction = st.number_input("SE reduction", min_value=1, value=4, step=1)
    se_activation = st.selectbox("SE activation", ["relu", "prelu", "silu", "gelu"], index=0)
    se_gate = st.selectbox("SE gate", ["sigmoid", "hardsigmoid"], index=0)
    use_denoiser = st.checkbox("Use trained DenoiseHead", value=True)

    st.header("Exposure Gate")
    use_gate = st.checkbox("Enable exposure gating", value=True,
                           help="Fades out enhancement strength as input brightness rises.")
    gate_low = st.slider("Gate low threshold", min_value=0.0, max_value=0.5, value=0.15, step=0.01,
                         help="Images at or below this brightness receive 100% enhancement.")
    gate_high = st.slider("Gate high threshold", min_value=0.2, max_value=1.0, value=0.55, step=0.01,
                          help="Images at or above this brightness receive 0% enhancement.")
    if gate_low >= gate_high:
        st.warning("Gate Low must be lower than Gate High.")

    st.header("Exposure ceiling")
    max_exposure = st.slider("MAX_EXPOSURE", min_value=0.80, max_value=1.00, value=0.95, step=0.01)
    knee = st.slider("knee", min_value=0.02, max_value=0.30, value=0.10, step=0.01)

uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    try:
        model, denoise_head, device = load_model(
            checkpoint_path, se_positions, reduction, se_activation, se_gate, use_denoiser
        )
    except FileNotFoundError:
        st.error(f"Checkpoint not found at '{checkpoint_path}'. "
                 f"Put the .pth file next to app.py, or fix the path in the sidebar.")
        st.stop()
    except RuntimeError as e:
        st.error("Failed to load checkpoint into the model — the SE config in the "
                 "sidebar probably doesn't match what this checkpoint was trained with.\n\n"
                 f"Details: {e}")
        st.stop()

    image = Image.open(uploaded)
    with st.spinner("Enhancing..."):
        enhanced_np, gate_val = run_inference(
            image, model, denoise_head, device, max_exposure, knee,
            use_gate=use_gate, gate_low=gate_low, gate_high=gate_high
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)
        in_mean = np.array(image.convert('RGB')).mean() / 255
        st.caption(f"Mean brightness: {in_mean:.3f}")
        if use_gate:
            st.caption(f"Gate scale factor applied: `{gate_val * 100:.1f}%`")
    with col2:
        st.subheader("Enhanced")
        st.image(enhanced_np, use_container_width=True)
        st.caption(f"Mean brightness: {enhanced_np.mean() / 255:.3f}")

    buf = io.BytesIO()
    Image.fromarray(enhanced_np).save(buf, format="PNG")
    st.download_button("Download enhanced image", data=buf.getvalue(),
                        file_name="enhanced.png", mime="image/png")
else:
    st.info("Upload a photo to test the model. Works on both low-light and normal-exposure "
            "images — try a well-lit photo too, to check the exposure ceiling is holding.")