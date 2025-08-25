# Export Model from Google Colab to DeepGuard

## Step 1: Export from Colab (Add to your notebook)

```python
# Save model for web deployment
import tensorflowjs as tfjs

# After training your model
model.save('deepfake_model.h5')

# Convert to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'web_model')

# Download files
from google.colab import files
import shutil

# Zip the model files
shutil.make_archive('deepfake_model', 'zip', 'web_model')
files.download('deepfake_model.zip')
```

## Step 2: Extract to DeepGuard folder

1. Download the zip file from Colab
2. Extract to DeepGuard folder
3. You should have:
   - `model.json`
   - `group1-shard1of1.bin` (or similar)

## Step 3: Update DeepGuard model path

```javascript
// In scan.html, change this line:
const loaded = await window.aiModel.loadModel('./web_model/model.json');
```

## Step 4: Adjust preprocessing (if needed)

Update `model-integration.js` based on your model:
- Input size (224x224, 256x256, etc.)
- Normalization range (0-1 or -1 to 1)
- Color format (RGB/BGR)

## Alternative: Use model URL

```javascript
// Load directly from Google Drive or GitHub
const loaded = await window.aiModel.loadModel('https://your-model-url/model.json');
```