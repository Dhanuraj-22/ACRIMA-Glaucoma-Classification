
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms
from pathlib import Path

# ---------- Configuration ----------
st.set_page_config(
    page_title="ACRIMA Glaucoma Detection",
    page_icon="👁️",
    layout="centered"
)

DEVICE = torch.device("cpu")
CHECKPOINT = Path(__file__).parent / "convnext_mlff.pth"

# ---------- Model architecture ----------
class ConvNeXt8020Fusion(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3)
        )

        channels = self.backbone.feature_info.channels()

        self.conv1 = nn.Conv2d(channels[0], 128, kernel_size=1)
        self.conv2 = nn.Conv2d(channels[1], 128, kernel_size=1)
        self.conv3 = nn.Conv2d(channels[2], 128, kernel_size=1)

        self.fusion = nn.Sequential(
            nn.Conv2d(128 * 3, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)

        f1 = self.conv1(features[0])
        f2 = self.conv2(features[1])
        f3 = self.conv3(features[2])

        target_size = f3.shape[2:]
        f1 = F.interpolate(
            f1, size=target_size,
            mode="bilinear", align_corners=False
        )
        f2 = F.interpolate(
            f2, size=target_size,
            mode="bilinear", align_corners=False
        )

        fused = torch.cat([f1, f2, f3], dim=1)
        fused = self.fusion(fused)
        fused = F.adaptive_avg_pool2d(fused, 1)
        fused = torch.flatten(fused, 1)

        return self.classifier(fused)


# ---------- Image preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT.name}"
        )

    model = ConvNeXt8020Fusion(num_classes=2)

    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    return model


# ---------- User interface ----------
st.title("👁️ Glaucoma Detection")
st.write(
    "Upload a retinal fundus image for an experimental "
    "AI-based glaucoma classification."
)

uploaded_file = st.file_uploader(
    "Choose a fundus image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded fundus image",
                 use_container_width=True)

        if st.button("Predict", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_model()
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]

                predicted_class = int(torch.argmax(probabilities))
                confidence = float(probabilities[predicted_class]) * 100

            # IMPORTANT:
            # Confirm this mapping matches your training dataset:
            # 0 = Normal, 1 = Glaucoma.
            labels = {0: "Normal", 1: "Glaucoma"}
            st.subheader(f"Prediction: {labels[predicted_class]}")
            st.write(f"Model confidence: {confidence:.2f}%")

            st.caption(
                "Research demonstration only. This result is not "
                "a medical diagnosis or a substitute for an eye exam."
            )

    except Exception as e:
        st.error(f"Could not run prediction: {e}")

