# 🚗 Smart Road Management

A complete, GUI-based toolkit for automated road image processing, detection, categorization, and management. Designed for projects involving road safety analysis, infrastructure monitoring, and intelligent reporting.

---

## 📁 Recommended Folder Structure

```
/videos
    road_clip1.mp4
/extracted_frames
/processed_images
    preprocessed
    postprocessed
/predictions
    predict1
    predict2
    predict3
    round1_separate
    round2_separate
    round3_no_detection
    merge_predictions
/originals
/gps_data.csv
/location_wise_data.csv
/location_groups
/summary_reports.csv
```

---

## 🚀 Features & Workflow

### 1️⃣ Frame Extraction

* Extracts keyframes from road videos using histogram-based scene detection.
* User-adjustable histogram threshold and frame interval for frame extraction.

### 2️⃣ Image Preprocessing

* Enhances extracted frames by adjusting:

  * Contrast (Alpha)
  * Brightness (Beta)
  * Sharpening
* Prepares better quality images for detection.

### 3️⃣ Prediction & Sorting (Multi-Round YOLO Detection)

* **Round 1:** Predict on preprocessed images.

  * Low-confidence detections separated.
  * No-detection images separated.
* **Post-process:** Enhance low-confidence images.
* **Round 2:** Predict on post-processed images.

  * Remaining no-detection images separated.
* **Round 3:** Predict on original no-detection images (without processing).
* Merges all final predictions for unified output.

### 4️⃣ Post-Processing Low Confidence Images

* Enhances low-confidence images with sharpening and optional blur.
* Increases chances of successful re-prediction.

### 5️⃣ Merge Final Predictions

* Combines labels and images from all prediction rounds.
* Creates unified label and image folders for downstream tasks.

### 6️⃣ Copy Predicted Originals (For GPS Extraction)

* Copies original images corresponding to successful detections.
* Ensures GPS & timestamp extraction is based on unaltered images.

### 7️⃣ GPS & Timestamp Extraction (OCR-based)

* Uses OCR to extract timestamps and GPS coordinates from image overlays.
* Generates a CSV: `gps_data.csv` with:

  * Filename
  * Timestamp
  * Latitude
  * Longitude

### 8️⃣ Group Images by Location (DBSCAN Clustering)

* Clusters images based on GPS coordinates.
* Creates structured folders for each location.
* Outputs:

  * `location_groups` folder with grouped images
  * `location_wise_data.csv` with location details

### 9️⃣ Categorize by Labels within Locations

* Categorizes images inside each location group based on YOLO annotations.
* Organizes images by damage type, road conditions, etc., as per label classes.

### 🔟 Generate Summary Report

* Generates `summary_reports.csv` with:

  * Location-wise image counts
  * Category breakdowns
  * Useful statistics for reporting

### 🔧 Manual Image Labeling Tool

* GUI-based tool for drawing bounding boxes & assigning labels.
* Supports:

  * Class selection
  * Saving annotations
  * Editing previous selections

---

## 🛠️ Tech Stack

* Python
* Gradio (User Interface)
* OpenCV, PIL, NumPy (Image Processing)
* Tesseract OCR
* YOLO (Object Detection)
* DBSCAN (Location Clustering)
* Geopy (Reverse Geocoding, optional)

---

## ⚡ How to Use

1. Launch the `Smart Road Management` script.
2. Follow the UI tabs sequentially:

   * Extract Frames
   * Preprocess Images
   * Run Predictions (3 rounds)
   * Post-process Low Confidence Images
   * Merge Predictions
   * Extract GPS & Timestamp
   * Group by Location
   * Categorize by Labels
   * Generate Summary
   * Use Manual Labeling (if required)
3. Use provided Browse buttons to select folders and files.
4. Adjust parameters as needed for your dataset.
5. Maintain recommended folder structure for organized outputs.

---

## 💡 Notes & Best Practices

* Preprocessing improves detection accuracy in poor lighting or low-quality frames.
* Multi-round predictions with separation logic ensure maximum detection coverage.
* Original frames used for GPS extraction to retain overlay data.
* Grouping & categorization streamline review and analysis.
* Manual labeling supports dataset correction and expansion.

---

## 📂 CSV Outputs Explained

| CSV File                 | Description                                 |
| ------------------------ | ------------------------------------------- |
| `gps_data.csv`           | Filename, Timestamp, Latitude, Longitude    |
| `location_wise_data.csv` | Clustered location details with image count |
| `summary_reports.csv`    | Category-wise image stats per location      |

---

## 🌟 Future Improvements

* Auto-enhancement loop for low-confidence images
* Interactive map visualization of location clusters
* Enhanced reporting with image thumbnails

---

## 👩‍💻 Developed By - **Mergu Sneha Krishnahari**
