import cv2
import numpy as np

# ── Disease information ───────────────────────────────────────────────────────
DISEASE_INFO = {
    "Early Blight": {
        "description": (
            "Early Blight is a fungal disease caused by Alternaria solani. "
            "It typically starts on older, lower leaves and moves upward. "
            "Characteristic dark brown spots with concentric rings (target-board pattern) appear on leaves."
        ),
        "causes": (
            "Caused by the fungus Alternaria solani. Spreads through infected plant debris, "
            "soil, and water splashes. Favored by warm temperatures (24–29°C) and high humidity."
        ),
        "prevention": [
            "Use certified disease-free seeds.",
            "Practice crop rotation (avoid planting tomatoes in the same spot each year).",
            "Remove and destroy infected leaves promptly.",
            "Apply copper-based or Mancozeb fungicide as a preventive measure.",
            "Avoid overhead irrigation; water at the base of the plant.",
            "Maintain adequate plant spacing for good air circulation.",
        ],
    },
    "Late Blight": {
        "description": (
            "Late Blight is caused by the water mold Phytophthora infestans — the same pathogen "
            "responsible for the Irish Potato Famine. It causes large, irregular, water-soaked lesions "
            "on leaves that turn brown and necrotic rapidly. It can destroy an entire crop within days."
        ),
        "causes": (
            "Caused by Phytophthora infestans. Spreads rapidly in cool (10–20°C), wet, and humid conditions. "
            "Spores spread through wind, rain, and infected plant material."
        ),
        "prevention": [
            "Plant resistant tomato varieties when available.",
            "Apply Metalaxyl or Cymoxanil-based fungicides preventively.",
            "Avoid working in the field when foliage is wet.",
            "Remove and destroy infected plants immediately — do not compost.",
            "Ensure good field drainage to reduce humidity.",
            "Monitor weather forecasts; treat before wet spells.",
        ],
    },
    "Septoria Leaf Spot": {
        "description": (
            "Septoria Leaf Spot is caused by the fungus Septoria lycopersici. "
            "It appears as small circular spots with dark borders and gray-white centers, "
            "usually starting on lower leaves. Severely infected leaves turn yellow and drop."
        ),
        "causes": (
            "Caused by Septoria lycopersici. Overwinters in infected plant debris in the soil. "
            "Spread by water splashes, insects, and tools. Favored by warm, wet, and humid weather."
        ),
        "prevention": [
            "Remove and destroy infected leaves and plant debris.",
            "Apply Chlorothalonil or Mancozeb fungicide every 7–10 days during wet periods.",
            "Mulch around plants to prevent soil splash onto lower leaves.",
            "Practice crop rotation (3-year cycle).",
            "Stake plants to improve airflow and reduce humidity.",
            "Avoid touching plants when wet.",
        ],
    },
    "Healthy": {
        "description": "The leaf appears healthy with no visible signs of disease or infection.",
        "causes": "No disease detected.",
        "prevention": [
            "Continue regular monitoring (inspect plants twice a week).",
            "Maintain good soil nutrition and balanced fertilization.",
            "Water consistently at the base; avoid wetting foliage.",
            "Keep the field free of weeds that can harbour pests and fungi.",
        ],
    },
}

# ── Recommendations ───────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    ("Early Blight", "Mild"): (
        "Remove affected leaves. Apply copper-based fungicide. "
        "Avoid overhead watering. Monitor closely."
    ),
    ("Early Blight", "Moderate"): (
        "Apply Mancozeb or Chlorothalonil every 7–10 days. "
        "Remove lower infected leaves. Improve air circulation."
    ),
    ("Early Blight", "Severe"): (
        "Apply systemic fungicide (Azoxystrobin). Consider removing heavily infected plants. "
        "Consult your local agricultural officer."
    ),
    ("Late Blight", "Mild"): (
        "Apply preventive copper fungicide immediately. "
        "Avoid wet foliage conditions. Increase monitoring frequency."
    ),
    ("Late Blight", "Moderate"): (
        "Apply Metalaxyl + Mancozeb combination. "
        "Remove and destroy infected parts. Increase plant spacing."
    ),
    ("Late Blight", "Severe"): (
        "Apply Dimethomorph or Cymoxanil urgently. "
        "Isolate the infected area. Seek agricultural extension advice."
    ),
    ("Septoria Leaf Spot", "Mild"): (
        "Remove lower infected leaves. Apply Chlorothalonil. Mulch around the base of plants."
    ),
    ("Septoria Leaf Spot", "Moderate"): (
        "Apply Mancozeb or copper fungicide every 7–10 days. "
        "Avoid leaf wetting during irrigation."
    ),
    ("Septoria Leaf Spot", "Severe"): (
        "Apply systemic fungicide. Remove severely infected plants. Improve drainage."
    ),
    ("Healthy", "None"): (
        "No treatment needed. Continue regular monitoring and good agricultural practices."
    ),
}


# ── Severity estimation ───────────────────────────────────────────────────────
def estimate_severity(image_rgb):
    """
    Returns (severity_label, infected_pct).
    severity_label: 'Mild' | 'Moderate' | 'Severe' | 'None'
    infected_pct: float 0–100
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # Green range in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    total_pixels = green_mask.size
    green_pixels = int(np.sum(green_mask > 0))
    infected_pixels = total_pixels - green_pixels
    infected_pct = (infected_pixels / total_pixels) * 100

    if infected_pct < 20:
        label = "Mild"
    elif infected_pct < 50:
        label = "Moderate"
    else:
        label = "Severe"

    return label, round(infected_pct, 1)


# ── Image quality check ───────────────────────────────────────────────────────
BLUR_THRESHOLD = 100
DARK_THRESHOLD = 50
BRIGHT_THRESHOLD = 220

def check_image_quality(image_rgb):
    """
    Returns (is_ok: bool, warnings: list[str])
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    warnings = []

    # Blur
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < BLUR_THRESHOLD:
        warnings.append(f"Image appears blurry (sharpness score: {lap_var:.1f}). Try a clearer photo.")

    # Brightness
    mean_brightness = float(np.mean(gray))
    if mean_brightness < DARK_THRESHOLD:
        warnings.append(f"Image is too dark (brightness: {mean_brightness:.1f}/255). Take photo in better lighting.")
    elif mean_brightness > BRIGHT_THRESHOLD:
        warnings.append(f"Image is overexposed (brightness: {mean_brightness:.1f}/255). Avoid direct strong light.")

    is_ok = len(warnings) == 0
    return is_ok, warnings


# ── Multi-image consensus ─────────────────────────────────────────────────────
def consensus_predict(predictions):
    """
    predictions: list of (class_name, confidence, probs)
    Returns (final_class, method_note)
    """
    from collections import Counter
    classes = [p[0] for p in predictions]
    counts = Counter(classes)
    majority, freq = counts.most_common(1)[0]

    if freq > 1:
        return majority, f"Majority vote ({freq}/{len(predictions)} images agree)"

    # No majority — pick highest confidence
    best = max(predictions, key=lambda x: x[1])
    return best[0], "No majority; selected highest confidence result"


# ── Recommendation lookup ─────────────────────────────────────────────────────
def get_recommendation(disease, severity):
    key = (disease, severity if disease != "Healthy" else "None")
    return RECOMMENDATIONS.get(key, "Please consult a local agricultural expert.")


# ── Disease info ──────────────────────────────────────────────────────────────
def get_disease_info(disease):
    return DISEASE_INFO.get(disease, {
        "description": "No information available.",
        "causes": "Unknown.",
        "prevention": [],
    })
