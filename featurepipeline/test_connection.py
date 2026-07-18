import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

print("Project name from env:", os.getenv("HOPSWORKS_PROJECT_NAME"))
print("API key loaded:", bool(os.getenv("HOPSWORKS_API_KEY")))

project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

fs = project.get_feature_store()
mr = project.get_model_registry()

print(f"✅ Connected to project: {project.name}")