import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models import GatedABMIL # Assuming you separated the model class

def generate_patient_heatmap(patient_id, model_path, embed_path, patch_dir, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the tuned model
    checkpoint = torch.load(model_path, map_location=device)
    model = GatedABMIL(embed_dim=checkpoint["embed_dim"], 
                       attn_dim=checkpoint["params"]["attn_dim"], 
                       attn_dropout=0.0) # Dropout off for inference
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    # 2. Load the patient's feature bag
    bag_features = torch.load(embed_path).to(device)
    
    # 3. Extract the attention weights
    with torch.no_grad():
        # Bypass the classifier heads, we only want the 'A' tensor
        _, A = model(bag_features)
    
    # Move attention array to CPU and normalize to 0.0 - 1.0
    attention_weights = A.squeeze().cpu().numpy()
    attention_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min() + 1e-8)
    
    # 4. Map weights to spatial coordinates
    # -> THIS REQUIRES KNOWING THE EXACT PATCH ORDER <-
    patch_files = get_ordered_patch_files(patch_dir, patient_id) 
    
    # 5. Reconstruct the WSI canvas
    max_x, max_y = find_max_coordinates(patch_files)
    canvas = np.ones((max_y + 256, max_x + 256, 3), dtype=np.uint8) * 255
    heatmap_overlay = np.zeros_like(canvas, dtype=np.float32)

    # 6. Paint the attention scores onto the canvas
    for i, patch_file in enumerate(patch_files):
        # Extract X, Y from your filename format (e.g., 972_0_512_256.jpg)
        _, _, x, y = parse_filename(patch_file)
        
        # Load the original RGB patch
        img = cv2.imread(str(patch_file))
        canvas[y:y+256, x:x+256] = img
        
        # Paint the corresponding attention weight block
        heatmap_overlay[y:y+256, x:x+256] = attention_weights[i]

    # 7. Apply Colormap and Blend
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_overlay), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Alpha blend: 60% original image, 40% heatmap
    blended = cv2.addWeighted(canvas, 0.6, heatmap_colored, 0.4, 0)

    # Save high-res output
    plt.imsave(output_path, blended)
    print(f"Heatmap successfully saved to {output_path}")