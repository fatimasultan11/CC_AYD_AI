**AYD-CC (Transcription)**
- AYD-CC is a Django-based transcription application that processes audio files and converts them into text using Whisper AI.
- The project focuses on efficient and accurate transcription.

**Features**
- Audio File Transcription: audio files and receive text transcriptions and summary (in progress).
- Django Framework: Built with Django for a scalable and reliable backend.
- Database Integration: Store transcription data and user configurations in a database.

**Prerequisites**
- Python 3.8
- Whisper AI (local setup)
- Django 3.2 or later
- PostgreSQL

**Set Up a Virtual Environment:**
- conda create --name AYD_CC_AI python=3.8
- conda activate AYD_CC_AI

**Install Dependencies:**
- pip install -r requirements.txt  
**Set Up Whisper AI:**
- Install Whisper locally by following the official setup guide: Whisper GitHub Repository.

<!-- Configure Environment Variables:
Create a .env file in the project root and add necessary variables: -->

<!-- plaintext
Copy code
DATABASE_URL=your_database_url  
SECRET_KEY=your_django_secret_key   -->
**Run Database Migrations:**
- python manage.py migrate  

**Start the Development Server:**
- python manage.py runserver  
- Access the Application:
- Open http://127.0.0.1:8000 in your browser.


**API Endpoints**
- http://127.0.0.1:8080/rabtai/# worksapp-rabt-AI
# worksapp-rabt-AI
# worksapp-rabt-AI
