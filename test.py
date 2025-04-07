from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import torch

def get_args_parser():
    parser = argparse.ArgumentParser(description="LangSAM Inference")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--video_path", type=str, help="Path to the input video")
    parser.add_argument("--text_prompt", type=str, required=True, help="Text prompt for segmentation")
    parser.add_argument("--output_path", type=str, default="output", help="Directory to save the output")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for image processing")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame for image processing")
    return parser

def ImageInference(image_pil, text_prompt, start_frame, end_frame):
    if start_frame is not None and end_frame is not None:
        if start_frame > end_frame:
            raise ValueError("start_frame must be less than or equal to end_frame")
        else:
            id = start_frame
            while id <= end_frame:
                image_pil = Image.open(f"{args.image_path}/frame_{id}.jpg").convert("RGB")
                image_np = np.array(image_pil)
                text = text_prompt
                model = LangSAM()
                results = model.predict([image_pil], [text])
                first_result = results[0]
                masks = first_result["masks"]
                try: 
                    if not masks:
                        overlay = image_np
                        print("No masks found for the given text prompt.")
                    
                except:
                    first_mask = masks[0]
                    first_mask=np.array(first_mask)
                    first_mask_3d = np.stack([first_mask] * 3, axis=-1)
                    overlay = np.where(first_mask_3d > 0, [255, 0, 0], image_np)
                    output_path = f"{args.output_path}/output_frame{id}.jpeg"
                    plt.imsave(output_path, overlay.astype(np.uint8))
                    print(f"Processed image saved to {output_path}")
                id += 1
                


    else:
        model = LangSAM()
        image_np = np.array(image_pil)
        text = text_prompt
        results = model.predict([image_pil], [text])
        first_result = results[0]
        masks = first_result["masks"]  # Alle Masken
        try: 
            if not masks:
                overlay = image_np
                print("No masks found for the given text prompt.")
        except:
            first_mask = masks[0]
            first_mask=np.array(first_mask)
            first_mask_3d = np.stack([first_mask] * 3, axis=-1)
            overlay = np.where(first_mask_3d > 0, [255, 0, 0], image_np)
        return overlay

def VideoInference(video_path, text_prompt, output_path):
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

        # Frame in PIL-Format konvertieren
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Inferenz durchführen
        with torch.no_grad():
            overlay = ImageInference(image_pil, text_prompt)
        # Overlay zurück in BGR konvertieren
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # Frame speichern
        out.write(overlay_bgr)
        del overlay_bgr, overlay, image_pil

        
        if count < 10:
            count += 1
        else:
            print("Clearing GPU cache...")
            torch.cuda.empty_cache()
            count = 0
            print(torch.cuda.memory_summary())

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
    if video_path:
        output_path = f"{output_path}/output_video.mp4"
    else:
        output_path = f"{output_path}/output_image.png"
    if video_path:
        VideoInference(video_path, text_prompt, output_path)
    elif start_frame is not None and end_frame is not None:
        overlay = ImageInference(image_path, text_prompt, start_frame, end_frame) 
    else:
        image_pil = Image.open(image_path).convert("RGB")
        overlay = ImageInference(image_pil, text_prompt)
        #plt.imsave(args.output_path, overlay.astype(np.uint8))
        #print(f"Processed image saved to {args.output_path}")