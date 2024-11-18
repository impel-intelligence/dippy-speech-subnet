# Voice Validation API Documentation

## Overview
This API provides endpoints for voice validation and management of voice samples. It allows users to submit voice recordings for validation, manage voice profiles, and retrieve validation results.

## Database Schemas

## Endpoints

### POST /telemetry_report
Reports telemetry data from clients.

**Headers:**
- `Git-Commit`: Git commit hash
- `Bittensor-Version`: Version of Bittensor
- `UID`: User identifier
- `Hotkey`: User's hot key
- `Coldkey`: User's cold key

**Response:**
- Status: 200 OK

### POST /event_report
Reports events from the system.

**Request Body:**

```json
{
"commit": "string",
"btversion": "string",
"uid": "string",
"hotkey": "string",
"coldkey": "string",
"payload": {},
"signature": {}
}
```


**Response:**
- Status: 200 OK
- Status: 400 Bad Request (if error occurs)

### POST /minerboard_update
Updates miner status on the minerboard.

**Request Body:**

```
json
{
"uid": "integer",
"hotkey": "string",
"hash": "string",
"block": "integer",
"admin_key": "string"
}
```



**Response:**
- Status: 200 OK
- Status: 403 Forbidden (if invalid admin key)

### GET /minerboard
Retrieves the current minerboard status.

**Response:**
- Status: 200 OK
- Body: Array of minerboard entries with leaderboard data

### POST /evaluate_model
Submits a model for evaluation.

**Headers:**
- `Git-Commit`: Git commit hash
- `Bittensor-Version`: Version of Bittensor
- `UID`: User identifier
- `Hotkey`: User's hot key
- `Coldkey`: User's cold key
- `signed_payload`: Signed payload data
- `miner_hotkey`: Miner's hot key

**Request Body:**


{
"hash": "string",
"repo_namespace": "string",
"repo_name": "string",
"config_template": "string",
"competition_id": "string",
"hotkey": "string"
}


**Response:**
- Status: 200 OK with evaluation results
- Status: 400 Bad Request if hash verification fails

### POST /get_or_create_model
Creates or retrieves a model entry.

**Request Body:**

json
{
"hash": "string",
"repo_namespace": "string",
"repo_name": "string",
"config_template": "string",
"competition_id": "string",
"hotkey": "string",
"admin_key": "string"
}


**Response:**
- Status: 200 OK with model entry
- Status: 400 Bad Request if hash verification fails
- Status: 403 Forbidden if invalid admin key

### GET /hc
Health check endpoint.

**Response:**
- Status: 200 OK