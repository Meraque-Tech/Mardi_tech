import os
import sys
import roboflow

rf = roboflow.Roboflow(api_key="BSHu0NfrIBFl58wtwWnP")
workspace = rf.workspace("rnd-kyodu")
folder_base = "MARDI_Phase_1_Pineapple_Pontian"

# Accept folder as CLI argument, fallback to ~/zed_extracted
if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = os.path.join(os.environ["HOME"], f"zed_extracted/{folder_base}")

if not os.path.isdir(folder):
    print(f"Error: folder not found: {folder}")
    sys.exit(1)

print(f"Uploading dataset from: {folder}")

workspace.upload_dataset(
    folder,
    "pineapple_ai_system",
    num_workers=10,
    project_license="MIT",
    project_type="instance-segmentation",
    batch_name=folder_base,
    num_retries=0,
    is_prediction=False,
)