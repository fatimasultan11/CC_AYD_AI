# FastAPI Data Analysis with Vanna AI

This project is a FastAPI application that performs data analysis on a PostgreSQL database using the Vanna AI package. It provides endpoints for setting database credentials, training the model, suggesting questions, and asking questions about the data.

## Features

- Set and save database credentials
- Train the Vanna AI model on your database
- Get AI-suggested questions about your data
- Ask questions and receive answers based on your database content

## Prerequisites

- Anaconda or Miniconda
- PostgreSQL database
- OpenAI API key

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. Create and activate the Conda environment:
   ```bash
   conda create -n aireport2 python=3.9
   conda activate aireport2
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. Activate the Conda environment:
   ```bash
   conda activate aireport2
   ```

2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

3. The API will be available at `http://localhost:8080/rabtai`.

## API Endpoints

### Set Database Credentials

```
POST /db_credentials
```
Request body:
```json
{
  "host": "localhost",
  "port": "5432",
  "database": "your_database",
  "user": "your_username",
  "password": "your_password"
}
```
```
