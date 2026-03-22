 Complete API Reference: UperNet + DINOv2 Backbone in HuggingFace Transformers

       1. Exact Class Names and Import Paths

       # Core classes
       from transformers import UperNetForSemanticSegmentation  # transformers.models.upernet.modeling_upernet
       from transformers import UperNetConfig                    # transformers.models.upernet.configuration_upernet
       from transformers import Dinov2Config                     # transformers.models.dinov2.configuration_dinov2
       from transformers import Dinov2Backbone                   # transformers.models.dinov2.modeling_dinov2
       from transformers import Dinov2Model                      # transformers.models.dinov2.modeling_dinov2
       from transformers import AutoBackbone                     # transformers.models.auto
       from transformers import AutoImageProcessor               # transformers.models.auto

       2. How to Create UperNetConfig with a DINOv2 Backbone Config

       The UperNetConfig constructor accepts backbone_config as its first parameter (a PreTrainedConfig or dict). You pass in a Dinov2Config with out_features set:

       from transformers import Dinov2Config, UperNetConfig, UperNetForSemanticSegmentation

       backbone_config = Dinov2Config(
           out_features=["stage3", "stage6", "stage9", "stage12"],  # For dinov2-base (12 layers)
           reshape_hidden_states=True,  # Default True - reshapes to 4D (B, C, H, W) for UperNet
           apply_layernorm=True,        # Default True - normalizes feature maps
       )

       config = UperNetConfig(
           backbone_config=backbone_config,
           num_labels=150,        # Number of segmentation classes (e.g., 150 for ADE20K)
           hidden_size=512,       # UperNet head hidden channels
       )

       model = UperNetForSemanticSegmentation(config)

       This creates a model with ALL RANDOM WEIGHTS -- both backbone and UperNet head.

       3. out_features / out_indices for DINOv2 Backbone

       DINOv2 stage names are generated dynamically from num_hidden_layers:

       # From configuration_dinov2.py line 100:
       self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, num_hidden_layers + 1)]

       For DINOv2-base (12 layers): ["stem", "stage1", "stage2", ..., "stage12"]
       For DINOv2-large (24 layers): ["stem", "stage1", "stage2", ..., "stage24"]

       UperNet requires exactly 4 feature maps (for FPN + PSP). You must pick 4 stages, typically evenly spaced:
       ┌──────────────┬───────────────────┬─────────────┬──────────────────────────────────────────────┐
       │    Model     │ num_hidden_layers │ hidden_size │           Recommended out_features           │
       ├──────────────┼───────────────────┼─────────────┼──────────────────────────────────────────────┤
       │ dinov2-base  │ 12                │ 768         │ ["stage3", "stage6", "stage9", "stage12"]    │
       ├──────────────┼───────────────────┼─────────────┼──────────────────────────────────────────────┤
       │ dinov2-large │ 24                │ 1024        │ ["stage6", "stage12", "stage18", "stage24"]  │
       ├──────────────┼───────────────────┼─────────────┼──────────────────────────────────────────────┤
       │ dinov2-giant │ 40                │ 1536        │ ["stage10", "stage20", "stage30", "stage40"] │
       └──────────────┴───────────────────┴─────────────┴──────────────────────────────────────────────┘
       You can also use out_indices instead of out_features:
       # Equivalent for dinov2-large:
       backbone_config = Dinov2Config(out_indices=[6, 12, 18, 24])

       Critical detail: Since DINOv2 is a ViT (isotropic architecture), ALL stages produce feature maps of the same hidden_size (e.g., 1024 for dinov2-large). This means
       self.backbone.channels will be [1024, 1024, 1024, 1024] -- all identical. The UperNet head handles this correctly.

       4. Initializing with Pretrained Backbone Weights but Random UperNet Head

       The load_backbone function in backbone_utils.py only uses AutoBackbone.from_config() (random weights), not from_pretrained(). To get pretrained backbone weights, you
       need a two-step approach:

       Method A: Load pretrained backbone, then build UperNet around it
       from transformers import AutoBackbone, UperNetConfig, UperNetForSemanticSegmentation

       # Step 1: Create config with backbone_config pointing to the pretrained model
       backbone_config = Dinov2Config.from_pretrained(
           "facebook/dinov2-large",
           out_features=["stage6", "stage12", "stage18", "stage24"],
           reshape_hidden_states=True,
       )

       config = UperNetConfig(
           backbone_config=backbone_config,
           num_labels=150,
           hidden_size=512,
       )

       # Step 2: Create model (random weights everywhere)
       model = UperNetForSemanticSegmentation(config)

       # Step 3: Load pretrained backbone weights into the backbone submodule
       pretrained_backbone = AutoBackbone.from_pretrained(
           "facebook/dinov2-large",
           out_features=["stage6", "stage12", "stage18", "stage24"],
       )
       model.backbone.load_state_dict(pretrained_backbone.state_dict())

       Method B: Use backbone kwarg (loads from repo)
       from transformers import UperNetConfig, UperNetForSemanticSegmentation

       # The consolidate_backbone_kwargs_to_config function in UperNetConfig.__init__
       # supports a `backbone` string that points to a HuggingFace repo.
       # When backbone_config is None and backbone is a valid repo, it fetches
       # the config from the repo.
       config = UperNetConfig(
           backbone="facebook/dinov2-large",
           backbone_config=None,
           num_labels=150,
           hidden_size=512,
       )
       # BUT: load_backbone() still calls AutoBackbone.from_config(), NOT from_pretrained()
       # So you still need to load weights manually as in Method A, Step 3.

       5. forward() Method Signature and Return Values

       Source file: /src/transformers/models/upernet/modeling_upernet.py line 293

       def forward(
           self,
           pixel_values: torch.Tensor | None = None,          # (B, 3, H, W)
           output_attentions: bool | None = None,
           output_hidden_states: bool | None = None,
           labels: torch.Tensor | None = None,                 # (B, H, W) - torch.LongTensor
           return_dict: bool | None = None,
           **kwargs,
       ) -> tuple | SemanticSegmenterOutput:

       Parameters:
       - pixel_values: torch.Tensor of shape (batch_size, num_channels, height, width) -- the input images.
       - labels: torch.LongTensor of shape (batch_size, height, width) -- ground truth segmentation maps. Indices in [0, ..., num_labels - 1]. The loss_ignore_index (default
       255) is used for pixels to ignore.

       Returns SemanticSegmenterOutput:
       - loss: torch.FloatTensor of shape (1,) -- returned when labels is provided. Cross-Entropy loss (main + auxiliary_loss_weight * auxiliary_loss).
       - logits: torch.FloatTensor of shape (batch_size, num_labels, height, width)
       - hidden_states: tuple of torch.FloatTensor, optional
       - attentions: tuple of torch.FloatTensor, optional

       6. How to Freeze Backbone Parameters

       # Freeze all backbone parameters
       for param in model.backbone.parameters():
           param.requires_grad = False

       # Verify only head parameters are trainable
       trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
       total_params = sum(p.numel() for p in model.parameters())
       print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")

       7. Output Logits Shape vs Input Labels Shape (Upsampling)

       UperNet DOES handle upsampling internally. From the source code (lines in forward()):

       # From modeling_upernet.py:
       logits = self.decode_head(features)
       logits = nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

       # Similarly for auxiliary head:
       auxiliary_logits = self.auxiliary_head(features)
       auxiliary_logits = nn.functional.interpolate(auxiliary_logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False)

       So:
       - Input pixel_values: (B, 3, H, W)
       - Output logits: (B, num_labels, H, W) -- same spatial size as input
       - Labels: (B, H, W) -- same spatial size as input

       The logits are bilinearly interpolated from the internal resolution back to the input resolution. The loss is computed with CrossEntropyLoss(ignore_index=255) directly
       between upsampled logits and labels.

       However, there is a documented warning: "The logits returned do not necessarily have the same size as the pixel_values passed as inputs. This is to avoid doing two
       interpolations and lose some quality when a user needs to resize the logits to the original image size as post-processing. You should always check your logits shape and
       resize as needed." -- But looking at the actual source code, it does interpolate to pixel_values.shape[2:]. The warning may be outdated or refer to edge cases.

       8. Known Issues / Considerations with DINOv2 + UperNet

       1. No official pre-trained DINOv2+UperNet models on HuggingFace. Facebook published dpt-dinov2-* models (DPT head, not UperNet) for depth estimation but NO
       upernet-dinov2-* models. There are no first-party facebook/dinov2-large-ade-upernet or similar model cards.
       2. Isotropic architecture mismatch. UperNet was designed for hierarchical backbones (Swin, ConvNeXt) that produce multi-scale features at different resolutions. DINOv2
       is a ViT -- all hidden states have the same spatial resolution (H/14 x W/14 for patch_size=14) and the same channel dimension. The FPN in UperNet becomes somewhat
       degenerate because all 4 feature maps are the same shape. It still works, but you lose the multi-scale benefit.
       3. The reshape_hidden_states=True is required. UperNet expects 4D feature maps (B, C, H, W). DINOv2 natively outputs 3D tensors (B, seq_len, hidden_size). The
       Dinov2Backbone reshapes them when this flag is True (stripping the CLS token and reshaping to spatial dimensions).
       4. All 4 backbone channel dimensions are identical. For dinov2-large, self.backbone.channels = [1024, 1024, 1024, 1024]. The UperNet head's lateral convolutions and FPN
       all get the same input channel dimension. This is fine but inefficient compared to hierarchical backbones.
       5. auxiliary_in_channels may need explicit setting. The auxiliary head defaults to using in_channels[in_index] where in_index=2. Since all channels are the same for
       DINOv2, this works automatically, but you may want to set auxiliary_in_channels explicitly in UperNetConfig if you change auxiliary head architecture.
       6. Memory with large images. DINOv2 default image_size=518 with patch_size=14 gives 37x37 = 1369 patches. At 1024 hidden_size (large), this is substantial. With
       UperNet's PPM and FPN on top, GPU memory usage is high.

       9. DINOv2 Model Variants (Config Details)


       ┌───────────────────────┬─────────────┬───────────────────┬─────────────────────┬────────────┬────────────┐
       │         Model         │ hidden_size │ num_hidden_layers │ num_attention_heads │ image_size │ patch_size │
       ├───────────────────────┼─────────────┼───────────────────┼─────────────────────┼────────────┼────────────┤
       │ facebook/dinov2-small │ 384         │ 12                │ 6                   │ 518        │ 14         │
       ├───────────────────────┼─────────────┼───────────────────┼─────────────────────┼────────────┼────────────┤
       │ facebook/dinov2-base  │ 768         │ 12                │ 12                  │ 518        │ 14         │
       ├───────────────────────┼─────────────┼───────────────────┼─────────────────────┼────────────┼────────────┤
       │ facebook/dinov2-large │ 1024        │ 24                │ 16                  │ 518        │ 14         │
       ├───────────────────────┼─────────────┼───────────────────┼─────────────────────┼────────────┼────────────┤
       │ facebook/dinov2-giant │ 1536        │ 40                │ 24                  │ 518        │ 14         │
       └───────────────────────┴─────────────┴───────────────────┴─────────────────────┴────────────┴────────────┘
       10. Complete Working Example

       import torch
       from transformers import (
           Dinov2Config,
           UperNetConfig,
           UperNetForSemanticSegmentation,
           AutoBackbone,
           AutoImageProcessor,
       )
       from PIL import Image

       NUM_LABELS = 150  # e.g. ADE20K

       # --- Step 1: Build config ---
       backbone_config = Dinov2Config.from_pretrained(
           "facebook/dinov2-large",
           out_features=["stage6", "stage12", "stage18", "stage24"],
           reshape_hidden_states=True,
           apply_layernorm=True,
       )

       config = UperNetConfig(
           backbone_config=backbone_config,
           num_labels=NUM_LABELS,
           hidden_size=512,
           use_auxiliary_head=True,
           auxiliary_loss_weight=0.4,
           auxiliary_channels=256,
           loss_ignore_index=255,
       )

       # --- Step 2: Create model with random weights ---
       model = UperNetForSemanticSegmentation(config)

       # --- Step 3: Load pretrained backbone weights ---
       pretrained_backbone = AutoBackbone.from_pretrained(
           "facebook/dinov2-large",
           out_features=["stage6", "stage12", "stage18", "stage24"],
       )
       model.backbone.load_state_dict(pretrained_backbone.state_dict())

       # --- Step 4: Freeze backbone ---
       for param in model.backbone.parameters():
           param.requires_grad = False

       # --- Step 5: Forward pass ---
       image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
       image = Image.new("RGB", (518, 518))
       inputs = image_processor(images=image, return_tensors="pt")

       # Inference
       with torch.no_grad():
           outputs = model(**inputs)
           logits = outputs.logits  # (1, 150, H, W)

       # Training with labels
       labels = torch.zeros(1, 518, 518, dtype=torch.long)  # Same spatial size as input
       outputs = model(pixel_values=inputs["pixel_values"], labels=labels)
       loss = outputs.loss      # scalar
       logits = outputs.logits  # (1, 150, 518, 518) -- upsampled to input size

       11. Key Source Files Referenced

       - UperNet modeling: https://github.com/huggingface/transformers/blob/main/src/transformers/models/upernet/modeling_upernet.py
       - UperNet config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/upernet/configuration_upernet.py
       - DINOv2 modeling: https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/modeling_dinov2.py
       - DINOv2 config: https://github.com/huggingface/transformers/blob/main/src/transformers/models/dinov2/configuration_dinov2.py
       - Backbone utilities: https://github.com/huggingface/transformers/blob/main/src/transformers/backbone_utils.py