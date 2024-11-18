import os
import requests
import dotenv
from datetime import datetime, timezone
from common.validation_utils import regenerate_hash
from scoring.common import EvaluateModelRequest
from datetime import datetime
# import pytest
from pg_persistence import Persistence, HashEntry

# Load environment variables from a .env file
dotenv.load_dotenv("../.env")

# Initialize Persistence with the PostgreSQL connection string
db = Persistence("postgresql://vapi:vapi@localhost:5432/vapi")

# Ensure migrations have been run
db.run_migrations()

# Define the model details
llm = "Manavshah/llama-test"

# Get API endpoint from environment variable with default
API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://localhost:9999")

def create_test_model_payload():
    """Factory function to create a consistent test model payload"""
    request_payload = {
        "repo_namespace": llm.split("/")[0],
        "repo_name": llm.split("/")[1],
        "config_template": "default",
        "hash": None,
        "revision": "main",
        "competition_id": "test",
        "hotkey": "example_hotkey",
        "block": 1,
        "tokenizer": "llama"
    }

    # Generate the hash
    request_payload["hash"] = str(
        regenerate_hash(
            request_payload["repo_namespace"],
            request_payload["repo_name"],
            request_payload["config_template"],
            request_payload["hotkey"]
        )
    )
    
    return request_payload

def test_ensure_model():
    """Function to test the /ensure_model endpoint"""
    request_payload = create_test_model_payload()
    print("Request payload:", request_payload)

    # Prepare headers with the admin key for authentication
    headers = {
        "Content-Type": "application/json",
        "admin-key": os.environ.get("ADMIN_KEY")
    }

    # Send a POST request to the /ensure_model endpoint with the headers
    response = requests.post(f"{API_ENDPOINT}/ensure_model", json=request_payload, headers=headers)

    # Check and print response status
    if response.status_code != 200:
        print("Error response:", response.status_code, response.text)
        return

    # Parse and print the response data
    try:
        response_data = response.json()
        print("Response data:", response_data)
    except ValueError as e:
        print("Error parsing response:", e)
        return

    # Validate the response type (adjust this as needed for your specific use case)
    expected_response_type = dict  # Change to 'str' or other type based on actual expected response

    assert isinstance(response_data, expected_response_type), "Response is not of the expected type."

    # Fetch the inserted hash entry from the database for verification
    db_entry = db.get_internal_result(request_payload["hash"])
    if db_entry:
        print("Internal database entry after test:", db_entry)
        
        # Assert that the database entry matches the expected content
        expected_db_entry = {
            'hash': request_payload["hash"],
            'total_score': 0.0,
            'alpha_score': None,
            'beta_score': None,
            'gamma_score': None,
            'notes': '',
            'repo_namespace': request_payload["repo_namespace"],
            'repo_name': request_payload["repo_name"],
            'timestamp': db_entry['timestamp'],  # Allow dynamic timestamp check
            'safetensors_hash': None,
            'status': 'QUEUED'
        }
        
        # Compare each key in the expected entry with the actual database entry
        for key, value in expected_db_entry.items():
            if db_entry[key] != value:
                print(f"ASSERTION FAILED: Mismatch for '{key}':")
                print(f"  Expected: {value}")
                print(f"  Found:    {db_entry[key]}")
                print("----")
                assert False, f"Validation failed for key '{key}'"

        print("SUCCESS: Database entry matches the expected contents.")

        # Remove the entry from the database after the test
        # removal_success = db.remove_record(request_payload["hash"])
        # if removal_success:
        #     print(f"Entry with hash '{request_payload['hash']}' was successfully removed from the database.")
        # else:
        #     print(f"Failed to remove entry with hash '{request_payload['hash']}' from the database.")
    else:
        print(f"No internal database entry found for the hash '{request_payload['hash']}'.")

    print("Response passed validation.")



def test_get_model_submission_details():
    """Function to test the /model_submission_details endpoint"""
    request_payload = create_test_model_payload()

    # First ensure the model exists by calling ensure_model
    headers = {
        "Content-Type": "application/json",
        "admin-key": os.environ.get("ADMIN_KEY")
    }

    # Now test the GET endpoint
    params = {
        "repo_namespace": request_payload["repo_namespace"],
        "repo_name": request_payload["repo_name"],
        "config_template": request_payload["config_template"],
        "hash": request_payload["hash"],
        "hotkey": request_payload["hotkey"]
    }
    
    response = requests.get(f"{API_ENDPOINT}/model_submission_details", 
                          params=params)

    # Validate response
    assert response.status_code == 200
    response_data = response.json()
    print(f"response_data : {response_data}")
    
    # Verify the response contains the expected data
    assert response_data["hash"] == request_payload["hash"]
    assert response_data["status"] == "QUEUED"
    
    print("Model submission details test passed successfully")



def test_get_next_model_to_eval(test_db: Persistence):
    
    
    # Act
    result = test_db.get_next_model_to_eval()

    # Assert
    assert result is not None
    assert result.hash == "hash1"  # Should get earliest QUEUED entry
    assert result.status == "QUEUED"
    



def test_minerboard_update():
    """Function to test the /minerboard_update endpoint"""
    # Prepare test payload
    request_payload = {
        "uid": 123,
        "hotkey": "example_hotkey",
        "hash": create_test_model_payload()["hash"],  # Reuse hash from existing test model
        "block": 12345
    }

    # Prepare headers with admin key for authentication
    headers = {
        "Content-Type": "application/json",
        "admin-key": os.environ.get("ADMIN_KEY")
    }

    # Send POST request to minerboard_update endpoint
    response = requests.post(
        f"{API_ENDPOINT}/minerboard_update", 
        json=request_payload, 
        headers=headers
    )

    # Validate response
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}: {response.text}"
    
    response_data = response.json()
    print("Minerboard update response:", response_data)

    # Verify the response contains expected data
    assert response_data["hash"] == request_payload["hash"]
    assert response_data["uid"] == request_payload["uid"]
    assert response_data["hotkey"] == request_payload["hotkey"]
    assert response_data["block"] == request_payload["block"]

    print("Minerboard update test passed successfully")

if __name__ == "__main__":
    # Run the test functions
    # test_ensure_model()
    # test_get_model_submission_details()
    test_minerboard_update()
    # test_get_next_model_to_eval(db)
