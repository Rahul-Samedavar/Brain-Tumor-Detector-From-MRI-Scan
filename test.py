from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path=r"Models\Xception",
    repo_id="Aurogenic/Brain-Tumour-Detector",
    repo_type="model"
)
