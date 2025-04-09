from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import torch
from lang_sam import utils

def get_args_parser():
    parser = argparse.ArgumentParser(description="LangSAM Inference")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--video_path", type=str, help="Path to the input video")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for segmentation")
    parser.add_argument("--output_path", type=str, default="output", help="Directory to save the output")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for image processing")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame for image processing")
    return parser

def combine_all_masks(image_np, masks):
    # Sicherstellen, dass alle Masken NumPy-Arrays sind
    masks = [np.array(mask) for mask in masks]

    # Erstelle eine leere Maske mit der gleichen Form wie das Eingangsbild
    combined_mask = np.zeros_like(image_np)

    # Iteriere über alle Masken und kombiniere sie
    for mask in masks:
        # Erweitere die Maske auf 3 Kanäle (RGB)
        mask_3d = np.stack([mask] * 3, axis=-1)

        # Wähle eine Farbe für die Maske (z. B. Rot)
        mask_color = [255, 0, 0]

        # Überlagere die Maske auf das kombinierte Bild
        combined_mask = np.where(mask_3d > 0, mask_color, combined_mask)

    # Kombiniere die Maske mit dem Eingangsbild
    overlay = np.where(combined_mask > 0, combined_mask, image_np)

    return overlay

def ImageInference(image_pil, text, start_frame, end_frame, model):
    if start_frame != 0 and end_frame is not None:
        if start_frame > end_frame:
            raise ValueError("start_frame must be less than or equal to end_frame")
        else:
            id = start_frame
            while id <= end_frame:
                image_pil = Image.open(f"{args.image_path}/frame_{id}.jpg").convert("RGB")
                image_np = np.array(image_pil)
                results = model.predict([image_pil], [text])
                first_result = results[0]
                masks = first_result["masks"]
                try: 
                    if not masks:
                        overlay = image_np
                        print("No masks found for the given text prompt.")
                    
                except:
                    #overlay = combine_all_masks(image_np, masks)
                    overlay = utils.draw_image(image_np, masks, first_result["mask_boxes"], first_result["mask_scores"], first_result["mask_labels"])
                    output_path = f"{args.output_path}/output_frame{id}.jpeg"
                    plt.imsave(output_path, overlay.astype(np.uint8))
                    print(f"Processed image saved to {output_path}")
                id += 1
                


    else:
        image_np = np.array(image_pil)
        results = model.predict([image_pil], [text])
        

        # Alle Keys aus Ergebnissen extrahieren
        first_result = results[0]
        masks = first_result["masks"]
        probs = first_result["scores"]
        labels = first_result["labels"]
        xyxy = first_result["boxes"]
          
        try: 
            if not masks:
                overlay = image_np
                print("No masks found for the given text prompt.")
        except:
            #overlay = combine_all_masks(image_np, masks)
            overlay = utils.draw_image(image_np, masks, xyxy, probs, labels)
        return overlay

def VideoInference(video_path, text_prompt, output_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Videoeigenschaften abrufen
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec für das Ausgabevideo
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if len(frame.shape) == 2:
            image_pil = Image.fromarray(frame, mode='L')
        else:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        with torch.no_grad():
            overlay = ImageInference(image_pil, text_prompt, start_frame, end_frame, model)
        
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        out.write(overlay_bgr)
        del overlay_bgr, overlay, image_pil

        
        if count < 10:
            count += 1
        else:
            print("Clearing GPU cache...")
            torch.cuda.empty_cache()
            count = 0


        print(f"counted:{count}")
        frame_count += 1
        print(f"Processed frame {frame_count}")

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    image_path = args.image_path
    text_prompt = args.text_prompt
    video_path = args.video_path
    output_path = args.output_path
    start_frame = args.start_frame
    end_frame = args.end_frame
    model = LangSAM()
    
    #Entscheidung Bild oder Video
    if video_path:
        output_path = f"{output_path}/output_video.mp4"
    else:
        output_path = f"{output_path}/output_image.png"
    
    #Entscheidung über Inferencemethode 
    if video_path:
        VideoInference(video_path, text_prompt, output_path, model)

    elif start_frame != 0 and end_frame is not None:
        overlay = ImageInference(image_path, text_prompt, start_frame, end_frame, model)

    else:
        image_pil = Image.open(image_path).convert("RGB")
        overlay = ImageInference(image_pil, text_prompt, start_frame, end_frame, model)