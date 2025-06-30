import os
import re
import cv2
import shutil
import pandas as pd
import numpy as np
import pytesseract
import gradio as gr
import gradio as gr
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont
from tkinter import filedialog
from ultralytics import YOLO 
from sklearn.cluster import DBSCAN
from geopy.geocoders import Nominatim
from collections import Counter

# ======================= CONFIG ======================= #
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

CATEGORY_MAP = {
    0: "pothole", 1: "manhole", 2: "cracks", 3: "surface damage",
    4: "patches", 5: "edge line", 6: "lane mark", 7: "lane divider",
    8: "warning lines", 9: "zebra crossing", 10: "sign board",
    11: "no entry zone", 12: "speed breakers"
}
PRIORITY_LABELS = ["surface damage", "pothole", "cracks", "patches", "manhole"]
VALID_EXTS = (".jpg", ".jpeg", ".png")
CLASSES = list(CATEGORY_MAP.values())

# ======================= Global State ======================= #
image_list, boxes = [], []
current_index, selected_class = 0, 0
temp_start, selected_folder, annotation_folder = None, "", ""

# ================== Logic ================== #
# ================== Frame Extraction ================== #
def extract_key_frames(video_path, output_dir, hist_thresh, frame_interval):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    prev_hist, saved, count = None, 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h = cv2.calcHist([gray], [0], None, [256], [0, 256])
            h = cv2.normalize(h, h).flatten()
            if prev_hist is None or cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL) < hist_thresh:
                cv2.imwrite(os.path.join(output_dir, f"frame{saved:1}.jpg"), frame)
                prev_hist = h
                saved += 1
        count += 1
    cap.release()
    return f"[INFO] Extracted {saved} frames."

# ================== Image Preprocessing ================== #
def preprocess_images(input_folder, output_folder, alpha, beta, sharpen):
    os.makedirs(output_folder, exist_ok=True)
    for f in os.listdir(input_folder):
        if f.lower().endswith(VALID_EXTS):
            img = cv2.imread(os.path.join(input_folder, f))
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            blur = cv2.GaussianBlur(img, (3, 3), 0)
            sharp = cv2.addWeighted(img, sharpen, blur, -0.5, 0)
            cv2.imwrite(os.path.join(output_folder, f), sharp)
    return "[INFO] Preprocessing complete."

# ================== Prediction and Sorting ================== #
def predict_and_sort(model_path, source_folder, project_folder, predict_name,
                     original_folder, conf_threshold, allow_low_conf, allow_no_det,
                     round_num, no_det_common_folder, no_det_final_folder):

    if round_num == "1":
        current_no_det_folder = no_det_common_folder
        current_low_conf_folder = no_det_common_folder
        separate_low_conf = True
    elif round_num == "2":
        current_no_det_folder = no_det_common_folder
        separate_low_conf = False
    elif round_num == "3":
        current_no_det_folder = no_det_final_folder
        separate_low_conf = False
    else:
        return "❌ Invalid round number."

    os.makedirs(current_no_det_folder, exist_ok=True)
    if separate_low_conf:
        os.makedirs(current_low_conf_folder, exist_ok=True)

    model = YOLO(model_path)
    res = model.predict(source=source_folder, save=True, save_txt=True,
                        project=project_folder, name=predict_name)
    
    base = os.path.join(project_folder, predict_name)

    for r in res:
        img_name = os.path.basename(r.path)
        pred_img = os.path.join(base, img_name)
        orig_img = os.path.join(original_folder, img_name)
        label_path = os.path.join(base, "labels", os.path.splitext(img_name)[0] + ".txt")

        if not os.path.exists(orig_img):
            continue

        if not r.boxes or len(r.boxes) == 0:
            if allow_no_det:
                shutil.copy(orig_img, os.path.join(current_no_det_folder, img_name))
            os.remove(pred_img)
            if os.path.exists(label_path):
                os.remove(label_path)

        else:
            confs = r.boxes.conf.cpu().numpy()
            if all(c < conf_threshold for c in confs) and allow_low_conf and separate_low_conf:
                shutil.copy(orig_img, os.path.join(current_low_conf_folder, img_name))
                os.remove(pred_img)
                if os.path.exists(label_path):
                    os.remove(label_path)

    return f"[INFO] Prediction & Sorting Round {round_num} complete."


# ================== Postprocessing Images ================== #
def postprocess_images(input_folder, output_folder, alpha=1.2, beta=20, sharpen=1.5, blur_ksize=5):
    os.makedirs(output_folder, exist_ok=True)
    for f in os.listdir(input_folder):
        if f.lower().endswith(VALID_EXTS):
            img = cv2.imread(os.path.join(input_folder, f))
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            blur = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
            sharp = cv2.addWeighted(img, sharpen, blur, -0.5, 0)
            cv2.imwrite(os.path.join(output_folder, f), sharp)
    return "[INFO] Post-processing images complete."

# ================== Merge Folders ================== #
def merge_final_predictions(round_folders, merged_labels, merged_images):
    """
    round_folders: List of 'predict' folders from each round
    merged_labels: Destination for merged label files
    merged_images: Destination for merged images
    """
    os.makedirs(merged_labels, exist_ok=True)
    os.makedirs(merged_images, exist_ok=True)

    for folder in round_folders:
        labels_folder = os.path.join(folder, "labels")

        # Merge Labels
        if os.path.exists(labels_folder):
            for label_file in os.listdir(labels_folder):
                if label_file.lower().endswith('.txt'):
                    src = os.path.join(labels_folder, label_file)
                    shutil.copy2(src, os.path.join(merged_labels, label_file))

        # Merge Images
        for f in os.listdir(folder):
            src = os.path.join(folder, f)
            if os.path.isfile(src) and f.lower().endswith(VALID_EXTS):
                shutil.copy2(src, os.path.join(merged_images, f))

    return f"[INFO] Merged predictions from {len(round_folders)} folders."


# ================== Copy Predicted Original frames for GPS Extraction ================== #
def copy_predicted_originals(prediction_folder, originals_folder, output_folder):
    """
    Copies originals to output_folder if their predicted version exists.
    """
    os.makedirs(output_folder, exist_ok=True)
    predicted_images = [f for f in os.listdir(prediction_folder) if f.lower().endswith(VALID_EXTS)]
    
    copied = 0
    for img in predicted_images:
        orig_path = os.path.join(originals_folder, img)
        if os.path.exists(orig_path):
            shutil.copy2(orig_path, os.path.join(output_folder, img))
            copied += 1
        else:
            print(f"[WARNING] Original not found: {img}")

    return f"[INFO] Copied {copied} original images."

# ================== GPS & Timestamp Extraction ================== #
def extract_gps_timestamp(folder_path, output_csv):
    """
    Extract Latitude, Longitude, and Timestamp from images using OCR.
    Saves results to CSV, appends if file exists.
    """
    data = []

    def numerical_sort(value):
        numbers = re.findall(r'\d+', value)
        return int(numbers[0]) if numbers else -1

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(VALID_EXTS)]
    files.sort(key=numerical_sort)

    for filename in files:
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Could not read image: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        pil_img = Image.fromarray(thresh)

        try:
            extracted_text = pytesseract.image_to_string(pil_img)
            extracted_text = re.sub(r'[^\x00-\x7F]+', ' ', extracted_text)

            lat_match = re.search(r'Lat(?:itude)?\s*[:\-]?\s*([-+]?\d{1,3}\.\d+)', extracted_text, re.IGNORECASE)
            lon_match = re.search(r'Long(?:itude)?\s*[:\-]?\s*([-+]?\d{1,3}\.\d+)', extracted_text, re.IGNORECASE) or \
                        re.search(r'Long\s+([-+]?\d{1,3}\.\d+)', extracted_text, re.IGNORECASE)
            time_match = re.search(r'(\d{2}[/\-| ]\d{2}[/\-| ]\d{2,4})\s*(\d{2}:\d{2}:\d{2})\s*([APap][Mm])', extracted_text)

            latitude = lat_match.group(1).strip() if lat_match else "Not found"
            longitude = lon_match.group(1).replace('°', '').strip() if lon_match else "Not found"
            timestamp = f"{time_match.group(1).replace('|', '/').replace('-', '/')} {time_match.group(2)} {time_match.group(3).upper()}" \
                .strip() if time_match else "Not found"

            print(f"✅ {filename} → Lat: {latitude}, Lon: {longitude}, Time: {timestamp}")

        except Exception as e:
            print(f"❌ OCR failed for {filename}: {e}")
            latitude, longitude, timestamp = "Not found", "Not found", "Not found"

        data.append({
            "Image": filename,
            "Latitude": latitude,
            "Longitude": longitude,
            "Timestamp": timestamp
        })

    df = pd.DataFrame(data)

    if os.path.exists(output_csv):
        df.to_csv(output_csv, mode='a', index=False, header=False)
        print(f"[INFO] Data appended to {output_csv}")
    else:
        df.to_csv(output_csv, index=False)
        print(f"[INFO] Data saved to {output_csv} (new file created)")

    return f"[INFO] Extraction complete for {len(data)} images."


# ================== Group Images by Location ================== #
# ========== Clean Folder Name ==========
def clean_folder_name(name):
    name = re.sub(r'[\\/:"*?<>|]+', '', name)
    return name.replace(' ', '_')[:100].strip()

# ========== Reverse Geocode with Fallback ==========
def get_location_name(geolocator, lat, lon):
    """
    Tries reverse geocoding, fallback to Lat_Lon label if fails.
    """
    try:
        location = geolocator.reverse((lat, lon), language='en', timeout=10)
        if location and location.raw.get('address'):
            for key in ['road', 'suburb', 'neighbourhood', 'village', 'town', 'city', 'county', 'state']:
                if key in location.raw['address']:
                    return location.raw['address'][key]
    except Exception as e:
        print(f"⚠️ Reverse geocoding failed at ({lat}, {lon}): {e}")

    return f"Lat_{round(lat, 5)}_Lon_{round(lon, 5)}"

# ========== Group Images by Location ==========
def group_images_by_location(csv_path, image_folder, output_base, distance_threshold_m=1000, output_csv=None):
    print("⏳ Loading data...")
    df = pd.read_csv(csv_path)

    df = df[(df['Latitude'].str.lower() != 'not found') & (df['Longitude'].str.lower() != 'not found')]
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)

    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)
        df['Sort_Key'] = df['Timestamp'].astype(str) + "_" + df['Image']
        df = df.sort_values(by='Sort_Key')
    else:
        df = df.sort_values(by='Image')

    coords = np.radians(df[['Latitude', 'Longitude']])
    eps = distance_threshold_m / 1000 / 6371.0088
    db = DBSCAN(eps=eps, min_samples=1, metric='haversine')
    df['Group'] = db.fit_predict(coords)

    print(f"✅ Found {len(set(df['Group']))} location groups.")

    geolocator = Nominatim(user_agent="geo_app")
    loc_count = {}
    cluster_summary = []

    for grp in sorted(set(df['Group'])):
        grp_df = df[df['Group'] == grp]
        lat, lon = grp_df.iloc[0][['Latitude', 'Longitude']]

        if pd.isna(lat) or pd.isna(lon):
            print(f"⚠️ Skipping group {grp} due to invalid coordinates.")
            continue

        loc_name = get_location_name(geolocator, lat, lon)
        loc_name = clean_folder_name(loc_name)
        loc_count[loc_name] = loc_count.get(loc_name, 0) + 1
        folder_name = f"{loc_name}_{loc_count[loc_name]}"

        folder_path = os.path.join(output_base, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        grp_images = []
        for img in grp_df['Image']:
            src = os.path.join(image_folder, img)
            dst = os.path.join(folder_path, img)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                grp_images.append(img)
            else:
                print(f"⚠️ Image not found: {src}")

        end_lat, end_lon = grp_df.iloc[-1][['Latitude', 'Longitude']]
        end_loc = get_location_name(geolocator, end_lat, end_lon)

        cluster_summary.append({
            "Group_ID": grp,
            "Folder_Name": folder_name,
            "Start_Coordinates": f"{lat}, {lon}",
            "End_Coordinates": f"{end_lat}, {end_lon}",
            "Start_Location": loc_name,
            "End_Location": end_loc,
            "Number_of_Images": len(grp_images)
        })

    if output_csv:
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            cluster_df = pd.DataFrame(cluster_summary)
            cluster_df.to_csv(output_csv, index=False)
            print(f"✅ Cluster summary saved to {output_csv}")
        except Exception as e:
            print(f"⚠️ Failed to save output CSV: {e}")

    print(f"🎉 Grouping complete. Images organized at: {output_base}")


# ================== Group Images by Category ================== #
def categorize_images_by_label(location_base, annotation_folder):
    """
    Categorizes images into folders based on their highest-priority detected label.
    """
    for loc_folder in os.listdir(location_base):
        loc_path = os.path.join(location_base, loc_folder)
        if not os.path.isdir(loc_path):
            continue

        for img in os.listdir(loc_path):
            if not img.lower().endswith(VALID_EXTS):
                continue

            lbl_path = os.path.join(annotation_folder, f"{os.path.splitext(img)[0]}.txt")
            if not os.path.exists(lbl_path):
                continue

            counter = Counter()
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        label = CATEGORY_MAP.get(cls, "unknown")
                        counter[label] += 1

            if not counter:
                continue

            # Priority based categorization
            top_label = max(counter, key=lambda l: (l in PRIORITY_LABELS, counter[l]))
            dst_folder = os.path.join(loc_path, top_label)
            os.makedirs(dst_folder, exist_ok=True)

            # Move image & Copy label
            shutil.move(os.path.join(loc_path, img), os.path.join(dst_folder, img))
            shutil.copy2(lbl_path, os.path.join(dst_folder, f"{os.path.splitext(img)[0]}.txt"))

    print("[INFO] Images and labels categorized within location folders.")

# ================== Generate Summary ================== #
def generate_label_summary(cluster_csv, location_folder, output_csv):
    """
    Generates label-wise counts for each location group and adds them to the cluster summary.

    cluster_csv: Path to existing cluster summary CSV (must contain 'Folder_Name')
    location_folder: Folder containing grouped images by location with category subfolders
    output_csv: Destination CSV with added label counts
    """
    label_names = list(CATEGORY_MAP.values())

    if not os.path.isfile(cluster_csv):
        print(f"❌ Cluster summary CSV not found: {cluster_csv}")
        return

    df = pd.read_csv(cluster_csv)

    if "Folder_Name" not in df.columns:
        print("❌ The CSV must contain 'Folder_Name' column.")
        return

    # Initialize label columns if missing
    for label in label_names:
        if label not in df.columns:
            df[label] = 0

    for idx, row in df.iterrows():
        folder_name = row["Folder_Name"]
        folder_path = os.path.join(location_folder, folder_name)
        label_counter = Counter()

        if not os.path.isdir(folder_path):
            print(f"⚠️ Location folder not found: {folder_path}")
            continue

        for category in os.listdir(folder_path):
            cat_folder = os.path.join(folder_path, category)
            if not os.path.isdir(cat_folder):
                continue

            for lbl_file in os.listdir(cat_folder):
                if lbl_file.lower().endswith(".txt"):
                    lbl_path = os.path.join(cat_folder, lbl_file)
                    with open(lbl_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                label_str = CATEGORY_MAP.get(class_id, "unknown")
                                label_counter[label_str] += 1

        # Update DataFrame
        for label in label_names:
            df.at[idx, label] = label_counter[label]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Label summary saved to: {output_csv}")

# ================== Manual Label ================== #
def update_display(img_path):
    global temp_start, boxes
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 100)  # Bigger size for clarity
    except:
        font = ImageFont.load_default()

    for x1, y1, x2, y2, cls in boxes:
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=4)

        label_text = f"{CLASSES[cls]}"

        bbox = font.getbbox(label_text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        text_bg_coords = [x_min, y_min - text_h - 6, x_min + text_w + 10, y_min]
        draw.rectangle(text_bg_coords, fill="red")

        draw.text((x_min + 5, y_min - text_h - 3), label_text, fill="white", font=font)

    if temp_start:
        draw.ellipse([temp_start[0] - 5, temp_start[1] - 5, temp_start[0] + 5, temp_start[1] + 5], fill="blue")

    return img



def set_point(evt: gr.SelectData):
    global temp_start, boxes
    if temp_start is None:
        temp_start = (evt.index[0], evt.index[1])
    else:
        x1, y1 = temp_start
        x2, y2 = evt.index[0], evt.index[1]
        boxes.append((x1, y1, x2, y2, selected_class))
        temp_start = None
    img = os.path.join(selected_folder, image_list[current_index])
    return update_display(img)


def save_annotations():
    global annotation_folder, image_list, current_index, selected_folder, boxes
    if not boxes:
        return "⚠️ Draw at least one box."

    img_path = os.path.join(selected_folder, image_list[current_index])
    img = Image.open(img_path)
    w, h = img.size
    label_path = os.path.join(annotation_folder, f"{os.path.splitext(image_list[current_index])[0]}.txt")
    labeled_img_path = os.path.join(annotation_folder, f"{os.path.splitext(image_list[current_index])[0]}_labeled.jpg")

    os.makedirs(annotation_folder, exist_ok=True)

    with open(label_path, "w") as f:
        for x1, y1, x2, y2, cls in boxes:
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    labeled_img = update_display(img_path)
    labeled_img.save(labeled_img_path)

    return f"✅ Saved {len(boxes)} boxes + labeled image for {image_list[current_index]}"


def next_image():
    global current_index, image_list, temp_start, boxes
    if current_index >= len(image_list) - 1:
        return None, "🎉 All images labeled."

    current_index += 1
    temp_start = None
    boxes.clear()
    img = os.path.join(selected_folder, image_list[current_index])
    return update_display(img), f"Image {current_index + 1}/{len(image_list)}"


def prev_image():
    global current_index, image_list, temp_start, boxes
    if current_index > 0:
        current_index -= 1
    temp_start = None
    boxes.clear()
    img = os.path.join(selected_folder, image_list[current_index])
    return update_display(img), f"Image {current_index + 1}/{len(image_list)}"


def refresh_image_list():
    global image_list
    image_list = [f for f in os.listdir(selected_folder) if f.lower().endswith(VALID_EXTS)]
    image_list.sort()


def set_selected_class(c):
    global selected_class
    selected_class = CLASSES.index(c)


def cancel_selection():
    global temp_start
    if temp_start:
        temp_start = None
        img = os.path.join(selected_folder, image_list[current_index])
        return update_display(img), "❌ Current selection cancelled."
    return None, "⚠️ No active selection to cancel."


def delete_last_box():
    global boxes
    if boxes:
        boxes.pop()
        img = os.path.join(selected_folder, image_list[current_index])
        return update_display(img), "🗑️ Last box deleted."
    return None, "⚠️ No boxes to delete."

# ================== Safe Browse Dialogs ================== #
def browse_folder_quick():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory()
    root.destroy()
    return folder

def browse_file_quick():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    root.destroy()
    return file

def browse_model_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file = filedialog.askopenfilename(filetypes=[("YOLO Model Files", "*.pt")])
    root.destroy()
    return file

def browse_save_csv():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return file

def browse_csv_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    root.destroy()
    return file


def load_images(folder):
    global image_list, selected_folder, annotation_folder, current_index, temp_start, boxes
    selected_folder = folder
    annotation_folder = os.path.join(folder, "labels")
    os.makedirs(annotation_folder, exist_ok=True)
    image_list = [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTS)]
    image_list.sort()
    current_index = 0
    temp_start = None
    boxes = []

    if not image_list:
        return None, "⚠️ No images found!"

    img = os.path.join(selected_folder, image_list[current_index])
    return update_display(img), f"Loaded {len(image_list)} images."


# ================== Gradio UI ================== #
with gr.Blocks(title="Fast Road Pipeline UI") as demo:
    gr.Markdown("# 🚀 Road Management Pipeline (Quick Browse + Working Controls)")
    
    # ================== Frame Extraction ================== #
    with gr.Tab("Frame Extraction"):
        video_path = gr.Textbox(label="Video File Path", interactive=True)  # Editable
        browse_video = gr.Button("📂 Browse Video")

        output_dir = gr.Textbox(label="Output Folder", interactive=True)  # Editable
        browse_output = gr.Button("📁 Browse Output Folder")

        hist = gr.Slider(0, 1, 0.6, label="Histogram Threshold")
        interval = gr.Number(label="Frame Interval", value=5)

        btn_extract = gr.Button("🎬 Extract Frames")
        output_msg = gr.Textbox(label="Status", interactive=False)

        browse_video.click(fn=browse_file_quick, outputs=video_path)
        browse_output.click(fn=browse_folder_quick, outputs=output_dir)

        btn_extract.click(
            extract_key_frames,
            inputs=[video_path, output_dir, hist, interval],
            outputs=output_msg
        )

    # ================== Image Precrocessing ================== #    
    with gr.Tab("Preprocessing"):
        input_folder = gr.Textbox(label="Input Folder", interactive=True)  # Editable after browse
        browse_input = gr.Button("📁 Browse Input Folder")

        output_folder = gr.Textbox(label="Output Folder", interactive=True)  # Editable after browse
        browse_output = gr.Button("📁 Browse Output Folder")

        alpha = gr.Slider(0.5, 3.0, 1.0, label="Contrast (Alpha)")
        beta = gr.Slider(0, 100, 20, label="Brightness (Beta)")
        sharpen = gr.Slider(0.0, 2.0, 1.5, label="Sharpening Strength")

        btn_preprocess = gr.Button("⚙️ Preprocess Images")
        output_msg = gr.Textbox(label="Status", interactive=False)

        browse_input.click(fn=browse_folder_quick, outputs=input_folder)
        browse_output.click(fn=browse_folder_quick, outputs=output_folder)

        btn_preprocess.click(
            preprocess_images,
            inputs=[input_folder, output_folder, alpha, beta, sharpen],
            outputs=output_msg
        )
        
    # ================== Prediction & Sorting ================== # 
    with gr.Tab("Prediction & Sorting"):
        model_path = gr.Textbox(label="YOLO Model Path", interactive=True)
        browse_model = gr.Button("📂 Browse YOLO Model (.pt)")

        source_folder = gr.Textbox(label="Source Image Folder", interactive=True)
        browse_source = gr.Button("📁 Browse Source Folder")

        project_folder = gr.Textbox(label="Prediction(Project) Folder", interactive=True)
        browse_project = gr.Button("📁 Browse Prediction(Project) Folder")

        predict_name = gr.Textbox(label="Predict Subfolder Name", value="predict")

        original_folder = gr.Textbox(label="Original Image Folder", interactive=True)
        browse_original = gr.Button("📁 Browse Original Folder")

        conf_threshold = gr.Slider(0.0, 1.0, 0.7, label="Confidence Threshold")
        allow_low_conf = gr.Checkbox(label="Allow Low Confidence Sorting", value=True)
        allow_no_det = gr.Checkbox(label="Allow No Detection Sorting", value=True)

        round_num = gr.Dropdown(choices=["1", "2", "3"], value="1", label="Round Number")

        no_det_common = gr.Textbox(label="Folder for separation of images after prediction (round1/2)", interactive=True)
        browse_no_det_common = gr.Button("📁 Browse Common Folder")

        no_det_final = gr.Textbox(label="No Detection Final Folder", interactive=True)
        browse_no_det_final = gr.Button("📁 Browse No Detection Final Folder")

        btn_predict = gr.Button("🚀 Predict & Sort")
        output_msg = gr.Textbox(label="Status", interactive=False)

        # Browse Button Actions
        browse_model.click(fn=browse_model_file, outputs=model_path)
        browse_source.click(fn=browse_folder_quick, outputs=source_folder)
        browse_project.click(fn=browse_folder_quick, outputs=project_folder)
        browse_original.click(fn=browse_folder_quick, outputs=original_folder)
        browse_no_det_common.click(fn=browse_folder_quick, outputs=no_det_common)
        browse_no_det_final.click(fn=browse_folder_quick, outputs=no_det_final)

        # Predict Action
        btn_predict.click(
            predict_and_sort,
            inputs=[
                model_path, source_folder, project_folder, predict_name,
                original_folder, conf_threshold, allow_low_conf, allow_no_det,
                round_num, no_det_common, no_det_final
            ],
            outputs=output_msg
        )

        
    # ================== Postprocessing Images ================== #
    with gr.Tab("Post-Processing Images"):
        input_folder = gr.Textbox(label="Input Folder", interactive=True)
        browse_input = gr.Button("📁 Browse Input Folder")

        output_folder = gr.Textbox(label="Output Folder", interactive=True)
        browse_output = gr.Button("📁 Browse Output Folder")

        alpha = gr.Slider(0.5, 3.0, 1.2, label="Contrast (Alpha)")
        beta = gr.Slider(0, 100, 20, label="Brightness (Beta)")
        sharpen = gr.Slider(0.0, 2.0, 1.5, label="Sharpening Strength")
        blur_ksize = gr.Slider(3, 11, 5, step=2, label="Blur Kernel Size (Odd Only)")

        btn_postprocess = gr.Button("⚙️ Post-Process Images")
        output_msg = gr.Textbox(label="Status", interactive=False)

        browse_input.click(fn=browse_folder_quick, outputs=input_folder)
        browse_output.click(fn=browse_folder_quick, outputs=output_folder)

        btn_postprocess.click(
            postprocess_images,
            inputs=[input_folder, output_folder, alpha, beta, sharpen, blur_ksize],
            outputs=output_msg
        )

    # ================== Merge Folders ================== #
    with gr.Tab("Merge Final Predictions"):
    
        round1 = gr.Textbox(label="Prediction Round 1 Folder", interactive=True)
        browse_r1 = gr.Button("📁 Browse Round 1 Folder")

        round2 = gr.Textbox(label="Prediction Round 2 Folder", interactive=True)
        browse_r2 = gr.Button("📁 Browse Round 2 Folder")

        round3 = gr.Textbox(label="Prediction Round 3 Folder", interactive=True)
        browse_r3 = gr.Button("📁 Browse Round 3 Folder")

        merged_labels = gr.Textbox(label="Merged Labels Folder", interactive=True)
        browse_labels = gr.Button("📁 Browse Labels Destination")

        merged_images = gr.Textbox(label="Merged Images Folder", interactive=True)
        browse_images = gr.Button("📁 Browse Images Destination")

        btn_merge = gr.Button("🗂️ Merge Final Predictions")
        merge_output = gr.Textbox(label="Status", interactive=False)

        # Browse actions
        browse_r1.click(fn=browse_folder_quick, outputs=round1)
        browse_r2.click(fn=browse_folder_quick, outputs=round2)
        browse_r3.click(fn=browse_folder_quick, outputs=round3)
        browse_labels.click(fn=browse_folder_quick, outputs=merged_labels)
        browse_images.click(fn=browse_folder_quick, outputs=merged_images)

        # Merge Action with lambda to combine inputs
        btn_merge.click(
            fn=lambda r1, r2, r3, labels, images: merge_final_predictions([r1, r2, r3], labels, images),
            inputs=[round1, round2, round3, merged_labels, merged_images],
            outputs=merge_output
        )

    # ================== Copy Predoicted Original frames for GPS Extraction ================== #
    with gr.Tab("Copy Predicted Originals"):

        pred_folder = gr.Textbox(label="Prediction Images Folder", interactive=True)
        browse_pred = gr.Button("📁 Browse Predictions Folder")

        orig_folder = gr.Textbox(label="Original Images Folder", interactive=True)
        browse_orig = gr.Button("📁 Browse Originals Folder")

        out_folder = gr.Textbox(label="Output Folder", interactive=True)
        browse_out = gr.Button("📁 Browse Output Folder")

        btn_copy = gr.Button("📂 Copy Originals")
        output_msg = gr.Textbox(label="Status", interactive=False)

        # Browse Actions
        browse_pred.click(fn=browse_folder_quick, outputs=pred_folder)
        browse_orig.click(fn=browse_folder_quick, outputs=orig_folder)
        browse_out.click(fn=browse_folder_quick, outputs=out_folder)

        # Copy Action
        btn_copy.click(
            copy_predicted_originals,
            inputs=[pred_folder, orig_folder, out_folder],
            outputs=output_msg
        )
        
    # ================== Extract GPS & Timestamp ================== #
    with gr.Tab("Extract GPS & Timestamp"):

        gps_input_folder = gr.Textbox(label="Image Folder", interactive=True)
        browse_gps_input = gr.Button("📁 Browse Image Folder")

        gps_output_folder = gr.Textbox(label="Output Folder", interactive=True)
        browse_gps_output_folder = gr.Button("📁 Browse Output Folder")

        gps_output_filename = gr.Textbox(label="Output CSV File Name (with .csv)", value="gps_data.csv")

        btn_extract_gps = gr.Button("🛰️ Extract GPS & Timestamp")
        gps_output_msg = gr.Textbox(label="Status", interactive=False)

        # Browse Actions
        browse_gps_input.click(fn=browse_folder_quick, outputs=gps_input_folder)
        browse_gps_output_folder.click(fn=browse_folder_quick, outputs=gps_output_folder)

        # Function to combine paths and run extraction
        def extract_gps_combined(image_folder, output_folder, output_filename):
            if not output_filename.lower().endswith(".csv"):
                return "❌ Output filename must end with .csv"

            if not os.path.isdir(image_folder):
                return "❌ Invalid Image Folder"

            if not os.path.isdir(output_folder):
                os.makedirs(output_folder, exist_ok=True)

            output_csv = os.path.join(output_folder, output_filename)
            return extract_gps_timestamp(image_folder, output_csv)

        # Extract Action
        btn_extract_gps.click(
            extract_gps_combined,
            inputs=[gps_input_folder, gps_output_folder, gps_output_filename],
            outputs=gps_output_msg
        )



    # ================== Group Images by Location ================== #
    with gr.Tab("Group by Location"):

        csv_path = gr.Textbox(label="CSV with Latitude & Longitude", interactive=True)
        browse_csv = gr.Button("📄 Browse CSV")

        image_folder = gr.Textbox(label="Image Source Folder", interactive=True)
        browse_images = gr.Button("📁 Browse Image Folder")

        output_base = gr.Textbox(label="Output Base Folder", interactive=True)
        browse_output = gr.Button("📁 Browse Output Folder")

        distance_thresh = gr.Number(label="Distance Threshold (meters)", value=1000)

        summary_csv = gr.Textbox(label="Cluster Summary Output CSV (optional)", interactive=True)
        browse_summary = gr.Button("📄 Browse Output CSV Location")

        btn_group = gr.Button("📦 Group Images by Location")
        group_msg = gr.Textbox(label="Status", interactive=False)

        # Browse actions
        browse_csv.click(fn=browse_csv_file, outputs=csv_path)
        browse_images.click(fn=browse_folder_quick, outputs=image_folder)
        browse_output.click(fn=browse_folder_quick, outputs=output_base)
        browse_summary.click(fn=browse_save_csv, outputs=summary_csv)

        # Grouping Action
        btn_group.click(
            lambda c, i, o, d, s: group_images_by_location(c, i, o, d, s),
            inputs=[csv_path, image_folder, output_base, distance_thresh, summary_csv],
            outputs=group_msg
        )

    # ================== Group Images by Category ================== #
    with gr.Tab("Categorize by Labels"):
        location_base = gr.Textbox(label="Location Base Folder", interactive=True)
        browse_loc = gr.Button("📁 Browse Location Folder")

        annotation_folder = gr.Textbox(label="Annotation (Labels) Folder", interactive=True)
        browse_ann = gr.Button("📁 Browse Annotation Folder")

        btn_categorize = gr.Button("🗂️ Categorize by Label")
        cat_msg = gr.Textbox(label="Status", interactive=False)

        browse_loc.click(fn=browse_folder_quick, outputs=location_base)
        browse_ann.click(fn=browse_folder_quick, outputs=annotation_folder)

        btn_categorize.click(
            lambda loc, ann: (categorize_images_by_label(loc, ann), "[INFO] Categorization complete."),
            inputs=[location_base, annotation_folder],
            outputs=cat_msg
        )

    # ================== Generate Summary ================== #
    with gr.Tab("Label Summary Report"):
        cluster_csv = gr.Textbox(label="Cluster Summary CSV", interactive=True)
        browse_cluster = gr.Button("📄 Browse Cluster CSV")

        location_folder = gr.Textbox(label="Location Group Folder", interactive=True)
        browse_location = gr.Button("📁 Browse Location Folder")

        output_csv = gr.Textbox(label="Output Summary CSV", interactive=True)
        browse_output = gr.Button("📄 Browse Output CSV Location")

        btn_summary = gr.Button("📊 Generate Label Summary")
        output_msg = gr.Textbox(label="Status", interactive=False)

        browse_cluster.click(fn=browse_csv_file, outputs=cluster_csv)
        browse_location.click(fn=browse_folder_quick, outputs=location_folder)
        browse_output.click(fn=browse_save_csv, outputs=output_csv)

        btn_summary.click(
            lambda clus, loc, out: (generate_label_summary(clus, loc, out), "[INFO] Summary generated."),
            inputs=[cluster_csv, location_folder, output_csv],
            outputs=output_msg
        )

    # ================== Manual Label ================== #
    with gr.Tab("Manual Image Labeling Tool"):


        folder_in = gr.Textbox(label="Image Folder", interactive=True)
        browse = gr.Button("📁 Browse")

        img_display = gr.Image(type="pil", height=500)
        status = gr.Textbox(label="Status", interactive=False)

        class_selector = gr.Dropdown(choices=CLASSES, label="Select Class", value=CLASSES[0])

        with gr.Row():
            btn_save = gr.Button("💾 Save Annotations")
            btn_prev = gr.Button("⬅️ Previous")
            btn_next = gr.Button("➡️ Next")

        with gr.Row():
            btn_cancel = gr.Button("❌ Cancel Current Selection")
            btn_delete = gr.Button("🗑️ Delete Last Box")

        browse.click(fn=browse_folder_quick, outputs=folder_in)
        folder_in.change(fn=load_images, inputs=folder_in, outputs=[img_display, status])

        class_selector.change(fn=set_selected_class, inputs=class_selector)

        img_display.select(fn=set_point, outputs=img_display)

        btn_save.click(fn=save_annotations, outputs=status)
        btn_prev.click(fn=prev_image, outputs=[img_display, status])
        btn_next.click(fn=next_image, outputs=[img_display, status])

        btn_cancel.click(fn=cancel_selection, outputs=[img_display, status])
        btn_delete.click(fn=delete_last_box, outputs=[img_display, status])
        
demo.launch()
