import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

print("Project name from env:", os.getenv("HOPSWORKS_PROJECT_NAME"))
print("API key loaded:", bool(os.getenv("HOPSWORKS_API_KEY")))

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
)

fs = project.get_feature_store()
mr = project.get_model_registry()

print(f"✅ Connected to project: {project.name}")