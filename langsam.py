from PIL import Image
from lang_sam import LangSAM
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import torch
from lang_sam import utils
import tools
import pandas as pd
import logging

def get_args_parser():
    parser = argparse.ArgumentParser(description="LangSAM Inference")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--video_path", type=str, default="./assets/Test.mp4", help="Path to the input video")
    parser.add_argument("--text_prompt", type=str, default="wheel.", help="Text prompt for segmentation")
    parser.add_argument("--output_path", type=str, default="output", help="Directory to save the output")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for image processing")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame for image processing")
    parser.add_argument("--confidence_threshold", type=float, default=0.45, help="Confidence threshold for predictions")
    parser.add_argument("--csv", type=bool, default=True, help="Save data to CSV")
    return parser


def ImageInference(image_pil, text, start_frame, end_frame, model, confidence_threshold, kalman)-> tuple:
    """
    Führt die Bild-Inferenz durch und verarbeitet die Ergebnisse.

    Args:
        image_pil (PIL.Image.Image): Eingabebild.
        text (str): Text-Prompt für die Segmentierung.
        start_frame (int): Start-Frame für die Verarbeitung.
        end_frame (int): End-Frame für die Verarbeitung.
        model (LangSAM): Das Modell für die Segmentierung.
        confidence_threshold (float): Schwellwert für die Konfidenz.
        kalman (object): Kalman-Filter-Objekt.

    Returns:
        tuple: Overlay-Bild, Scores, Labels, Bounding Boxes.
    """
    if start_frame != 0 and end_frame is not None:
        if start_frame > end_frame:
            raise ValueError("start_frame must be less than or equal to end_frame")
        else:
            id = start_frame
            while id <= end_frame:
                image_pil = Image.open(f"{args.image_path}/frame_{id}.jpg").convert("RGB")
                image_np = np.array(image_pil)
                results = model.predict([image_pil], [text], [0.45])
                first_result = results[0]
                masks = first_result["masks"]
                probs = first_result["scores"]
                labels = first_result["labels"]
                xyxy = first_result["boxes"]

                if masks is None or len(masks) == 0:
                        overlay = image_np
                        logging.info("No masks found for the given text prompt.")

                else:
                    #overlay = combine_all_masks(image_np, masks)
                    overlay = utils.draw_image(image_np, masks, xyxy, probs, labels)
                    output_path = f"{args.output_path}/output_frame{id}.jpeg"
                    plt.imsave(output_path, overlay.astype(np.uint8))
                    logging.info(f"Processed image saved to {output_path}")
                id += 1



    else:
        image_np = np.array(image_pil)
        results = model.predict([image_pil], [text])


        # Alle Keys aus Ergebnissen extrahieren
        first_result = results[0]
        masks = first_result["masks"]
        scores = first_result["scores"]
        labels = first_result["labels"]
        xyxy = first_result["boxes"]
        logging.debug(f"Results: {results}")
        logging.debug(f"Scores: {scores}")
        logging.debug(f"Labels: {labels}")
        logging.debug(f"Boxes: {xyxy}")


        if masks is None or len(masks) == 0:
            overlay = image_np
            logging.info("No masks found for the given text prompt.")

        else:
            # Filtere Scores, Boxen und Labels basierend auf dem Schwellwert
            valid_indices = scores >= confidence_threshold  # Nur Scores >= 0.45 berücksichtigen
            scores = scores[valid_indices]
            xyxy = xyxy[valid_indices]
            labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
            masks = masks[valid_indices]

        if len(scores) == 0:
            overlay = image_np
            logging.info("No valid detections after applying the threshold.")

        else:
            for index, label in enumerate(labels):
                if label == "car":
                    x_min = int(xyxy[index - 1][0])
                    y_min = int(xyxy[index - 1][1])
                    x_max = int(xyxy[index - 1][2])
                    y_max = int(xyxy[index - 1][3])
                    center_x = (x_max - x_min) / 2 + x_min
                    center_y = (y_max - y_min) / 2 + y_min
                    ROI = np.array([[center_x], [center_y]], np.float32)
                    kalman.correct(ROI)
                    prediction = tools.kalman_predict(kalman, ROI)
                    predicted_x = int(prediction[0][0])  # x-Koordinate
                    predicted_y = int(prediction[1][0])  # y-Koordinate
                else:
                    prediction = None

                if index == len(labels) - 1:
                    if prediction is not None:
                        #cv2.circle(image_np, (predicted_x, predicted_y), radius=5, color=(0, 255, 0), thickness=-1)
                        logging.info(f"Predicted position: ({predicted_x}, {predicted_y})")
                        overlay = utils.draw_image(image_np, masks, xyxy, scores, labels)
                    else:
                        overlay = utils.draw_image(image_np, masks, xyxy, scores, labels)

                else:
                    if prediction is not None:
                        #cv2.circle(image_np, (predicted_x, predicted_y), radius=5, color=(0, 255, 0), thickness=-1)
                        logging.info(f"Predicted position: ({predicted_x}, {predicted_y})")
                        overlay = utils.draw_image(image_np, masks, xyxy, scores, labels)
                    else:
                        overlay = utils.draw_image(image_np, masks, xyxy, scores, labels)
        return overlay, scores, labels, xyxy

def VideoInference(video_path, text_prompt, output_path, model, confidence_threshold, kalman)-> pd.DataFrame:
    """
    Führt die Video-Inferenz durch und speichert die Ergebnisse.

    Args:
        video_path (str): Pfad zum Eingabevideo.
        text_prompt (str): Text-Prompt für die Segmentierung.
        output_path (str): Pfad zum Ausgabevideo.
        model (LangSAM): Das Modell für die Segmentierung.
        confidence_threshold (float): Schwellwert für die Konfidenz.
        kalman (object): Kalman-Filter-Objekt.

    Returns:
        pd.DataFrame: DataFrame der Confidence Scores.
    """  
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
    iteration_id = 1
    confidence_data = []
    data = []
    x_min = []
    y_min = []
    x_max = []
    y_max = []

    while True and frame_count <= 54000:
        ret, frame = cap.read()

        if not ret:
            break

        if len(frame.shape) == 2:
            image_pil = Image.fromarray(frame, mode='L')
        else:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            overlay, scores, labels, xyxy = ImageInference(image_pil, text_prompt, start_frame, end_frame, model, confidence_threshold, kalman)

        if len(xyxy) > 0:
            x_min = [float(bbox[0]) for bbox in xyxy]
            y_min = [float(bbox[1]) for bbox in xyxy]
            x_max = [float(bbox[2]) for bbox in xyxy]
            y_max = [float(bbox[3]) for bbox in xyxy]
        else:
            logging.warning("Bounding box has unexpected length:", len(xyxy), xyxy)
            x_min, y_min, x_max, y_max = 0, 0, 0, 0
            labels = "n/a"
            scores = "n/a"

        data.append({
            "Iteration": iteration_id,
            "Label": labels,
            "Confidence Score": scores,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max
        })

        iteration_id += 1
        confidence_data.extend([(iteration_id, score) for score in scores])

        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)

        out.write(overlay_bgr)
        del overlay_bgr, overlay, image_pil


        if count < 10:
            count += 1
        else:
            logging.info("Clearing GPU cache...")
            torch.cuda.empty_cache()
            count = 0


        logging.info(f"counted:{count}")
        frame_count += 1
        logging.info(f"Processed frame {frame_count}")

    cap.release()
    out.release()
    logging.info(f"Video saved to {output_path}")

    df = pd.DataFrame(data)
    logging.debug(f"DataFrame: {df}")
    df.to_csv(f"{args.output_path}/lang_sam_wheel.csv", index=False)
    return df

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    image_path = args.image_path
    text_prompt = args.text_prompt
    video_path = args.video_path
    output_path = args.output_path
    start_frame = args.start_frame
    end_frame = args.end_frame
    confidence_threshold = args.confidence_threshold
    csv_bool = args.csv

    logging.basicConfig(
        filename="output.log",  
        level=logging.DEBUG,  
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Log-Level für die Konsole
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logging.getLogger().addHandler(console_handler)


    model = LangSAM()
    kalman = tools.kalman_init()

    #Entscheidung Bild oder Video
    if video_path:
        output_path = f"{output_path}/inference_video.mp4"
    else:
        output_path = f"{output_path}/output_image.png"

    #Entscheidung über Inferencemethode
    if video_path:
        data = VideoInference(video_path, text_prompt, output_path, model, confidence_threshold, kalman)
        

    elif start_frame != 0 and end_frame is not None:
        overlay = ImageInference(image_path, text_prompt, start_frame, end_frame, model, confidence_threshold, kalman)

    else:
        image_pil = Image.open(image_path).convert("RGB")
        overlay = ImageInference(image_pil, text_prompt, start_frame, end_frame, model, confidence_threshold, kalman)