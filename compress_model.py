import pickle
import joblib

# Step 1: Load the existing pickle model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Step 2: Save it with joblib compression
joblib.dump(model, 'model_compressed.pkl', compress=3)

print("Model compressed and saved as model_compressed.pkl")
