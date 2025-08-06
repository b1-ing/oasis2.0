import json

# Load the original JSON data
with open("posts/posts_NationalServiceSG.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Remove the 'virality_prediction' key from each object
for item in data:
    item.pop("virality_prediction", None)  # Safe removal

# Save to a new file (or overwrite original if preferred)
with open("posts/posts_NationalServiceSG.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("✅ Removed 'virality_prediction' column and saved to predictions_without_virality.json")
