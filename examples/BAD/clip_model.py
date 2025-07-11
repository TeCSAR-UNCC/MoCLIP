import my_clip
# import torch
# import os

# Previous Function
# def load_clip_model(args):
#     clip_model = my_clip.load("ViT-B/32", device=args.device, jit=False,
#                               download_root=os.path.join(os.environ.get("TORCH_HOME"), 'clip'))  # Must set jit=False for training
#     my_clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
#     clip_model.eval()
#     for p in clip_model.parameters():
#         p.requires_grad = False
#     return clip_model

import torch
import os
from my_clip.motion_clip import MotionTransformer, encode_motion  # Import your motion CLIP model
from my_clip.moCLIP import MotionTransformer as MotionTransformerv2

def load_clip_model(args):
    """ Load the Motion CLIP model instead of OpenAI's CLIP. """
    clip_model = my_clip.load("ViT-B/32", device=args.device, jit=False,
                              download_root=os.path.join(os.environ.get("TORCH_HOME"), 'clip'))  # Must set jit=False for training
    my_clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16

    print(args)

    # Replace their CLIP model with our MotionTransformer
    clip_model.motion = MotionTransformerv2("t2m").to(args.device)
    
    # Override the encode_motion method to use our implementation
    clip_model.encode_motion = encode_motion.__get__(
        clip_model, type(clip_model)
    )

    # Load your motion CLIP weights
    motion_clip_path = args.motion_clip  # Ensure this is correctly set in the config
    kpt = torch.load(motion_clip_path, map_location=args.device)
    clip_model.load_state_dict(kpt, strict=False)

    for p in clip_model.parameters():
        p.requires_grad = False
    
    print("LOADED MOTION CLIP MODEL")
    clip_model.to(args.device)
    
    return clip_model  # Return the modified model

