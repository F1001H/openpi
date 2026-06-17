import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class KoboInputs(transforms.DataTransformFn):
    """Converts live deployment inputs to the specific namespaced internal format."""
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1. Adapt and condition your camera frame safely to uint8 (H, W, C)
        base_image = _parse_image(data["observation/image"])
        base_image_external = _parse_image(data["observation/external_image"])

        # 2. Build the standard namespaced interface structure OpenPI mandates
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb":  base_image_external,
                # Pad out unused camera pipelines with matching resolution shapes
                "left_wrist_0_rgb": base_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        # Handle your text prompts and target action steps flatly
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class KoboOutputs(transforms.DataTransformFn):
    """Slices the model outputs back down to your 7 DOF task-space coordinate shape."""
    
    def __call__(self, data: dict) -> dict:
        # The model's raw internal forward pass outputs a (10, 32) matrix footprint.
        # Since your target robot moves via a 7 DOF task-space pose interface,
        # explicitly slice away indices 7 through 31 (the padding zeros).
        raw_actions = np.asarray(data["actions"], dtype=np.float32)
        
        return {"actions": raw_actions[..., :8]}