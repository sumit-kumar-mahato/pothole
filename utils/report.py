def generate_summary(detections):
    summary = {
        "pothole": 0,
        "longitudinal_crack": 0,
        "transverse_crack": 0,
        "alligator_crack": 0
    }

    for d in detections:
        summary[d["label"]] += 1

    total = sum(summary.values())

    if summary["pothole"] == 0:
        risk = "Safe"
    elif summary["pothole"] < 5:
        risk = "Moderate"
    else:
        risk = "High Risk"

    return summary, total, risk