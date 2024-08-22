import time
import requests

def check_model_status(model_url, headers):
    """
    Check if the model is ready or still loading.

    Args:
        model_url (str): The Hugging Face API URL for the model.
        headers (dict): The headers containing the authorization token.

    Returns:
        bool: True if the model is ready, False if it is still loading.
        int: Estimated time to load the model in seconds (if still loading).
    """
    response = requests.get(model_url, headers=headers)
    
    if response.status_code == 503:
        error_info = response.json()
        wait_time = error_info.get("estimated_time", 60)  # Default wait time if not provided
        return False, wait_time
    elif response.status_code == 200:
        return True, 0
    else:
        response.raise_for_status()

def wait_for_model(model_url, headers):
    """
    Wait for the model to load if it is currently unavailable.

    Args:
        model_url (str): The Hugging Face API URL for the model.
        headers (dict): The headers containing the authorization token.
    """
    while True:
        model_ready, wait_time = check_model_status(model_url, headers)
        if model_ready:
            print("Model is ready for use.")
            break
        else:
            print(f"Model is still loading. Retrying in {int(wait_time)} seconds...")
            time.sleep(wait_time)

# Example usage
huggingface_api_key = "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"  # Replace with your actual Hugging Face API key
headers = {
    "Authorization": f"Bearer {huggingface_api_key}",
    "Content-Type": "application/json"
}
image_model_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2-0.5B"

wait_for_model(image_model_url, headers)
