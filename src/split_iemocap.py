import json
from sklearn.model_selection import train_test_split

# Step 1: load original datasetï¼ˆiemocap_all.jsonï¼‰
with open("iemocap_all.json", "r") as f:
    data = json.load(f)

# Step 2: filter based on sessions
train_val_data = [item for item in data if item["session"] in ["Session1", "Session3", "Session4", "Session5"]]
test_data = [item for item in data if item["session"] == "Session2"]

# Step 3: split test and validation
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

# Step 4: sava as new json
with open("iemocap_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("iemocap_val.json", "w") as f:
    json.dump(val_data, f, indent=2)

with open("iemocap_test.json", "w") as f:
    json.dump(test_data, f, indent=2)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")