
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------

import os
from huggingface_hub import HfApi


# ---------------------------------------------------------
# Initialize Hugging Face API
# ---------------------------------------------------------

print("Connecting to Hugging Face...")

api = HfApi(token=os.getenv("HF_TOKEN"))


# ---------------------------------------------------------
# Upload Deployment Files to Hugging Face Space
# ---------------------------------------------------------

print("Uploading deployment files...")

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="vidyaa2026/Tourism_Package_Prediction",
    repo_type="space",
    path_in_repo=""
)

print("Deployment completed successfully.")
