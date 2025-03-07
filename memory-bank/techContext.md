# Technical Context

## Technology Stack

### Core Technologies
1. Python
   - Primary development language
   - Used across all services
   - Handles scoring and validation logic

2. Docker
   - Container orchestration
   - Service isolation
   - Deployment management
   - Multiple service Dockerfiles:
     - worker.Dockerfile
     - evaluator.Dockerfile
     - vapi.Dockerfile
     - modelq.Dockerfile

3. PostgreSQL
   - Persistence layer
   - Stores validation results
   - Manages miner registry
   - Handles migrations

## Development Setup

### Project Structure
```
dippy-speech-subnet/
├── scoring/               # Scoring system implementation
│   ├── scoring_logic/    # Core scoring algorithms
│   ├── prompt_templates/ # Model-specific templates
│   └── tests/           # Test suite
├── voice_validation_api/ # Validation API service
├── common/              # Shared utilities
├── neurons/             # Mining and validation
├── utilities/           # Helper functions
└── validator_updater/   # Validator maintenance
```

### Key Dependencies
1. Core Requirements
   - Listed in requirements.txt
   - Separate requirement files per service:
     - requirements.api.txt
     - requirements.eval.txt
     - requirements.miner.txt
     - requirements.validator.txt

2. Development Tools
   - pyproject.toml for project configuration
   - Docker Compose for local development
   - Migration scripts for database management

## Technical Constraints

### System Requirements
1. Processing
   - Efficient resource utilization
   - Parallel processing capabilities
   - Queue management system

2. Storage
   - PostgreSQL database
   - File system for temporary storage
   - Result persistence

3. Network
   - Inter-service communication
   - API endpoints
   - Queue management

### Performance Requirements
1. Scalability
   - Horizontal scaling through Docker
   - Queue-based load distribution
   - Resource optimization

2. Reliability
   - Error handling
   - Service recovery
   - Data consistency

3. Monitoring
   - Performance tracking
   - Resource utilization
   - System health checks

## Service Architecture

### Voice Validation API
- REST API endpoints
- Worker queue management
- PostgreSQL integration
- Service maintenance scripts

### Scoring System
- Multiple scoring methods
- Template-based processing
- Evaluation logic
- Result generation

### Distributed Processing
- Miner coordination
- Validator management
- Queue processing
- Resource allocation

## Development Workflow

### Local Development
1. Setup
   ```bash
   # Install dependencies
   pip install -r requirements.txt
   
   # Run services
   docker-compose up
   ```

2. Testing
   - Unit tests in scoring/tests/
   - API tests in voice_validation_api/test_api.py
   - Integration testing through Docker Compose

### Deployment
1. Container Build
   - Multiple Dockerfiles for different services
   - Compose files for orchestration:
     - docker-compose.yml
     - local-compose.yml
     - min_compute.yml

2. Service Management
   - Start/stop scripts
   - Maintenance procedures
   - Migration handling

## Monitoring and Maintenance

### System Health
1. Performance Monitoring
   - Resource utilization
   - Queue status
   - Processing metrics

2. Error Tracking
   - Logging system
   - Error reporting
   - Debug capabilities

3. Maintenance
   - Database migrations
   - Service updates
   - System cleanup

### Security Considerations
1. Data Protection
   - Secure storage
   - Access control
   - Input validation

2. Service Security
   - Container isolation
   - Network security
   - Error handling
