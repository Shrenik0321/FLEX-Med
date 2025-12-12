# FLEX-Med

## Backend Setup

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file and add your Supabase credentials. You can use the `.env.example` file as a template:
    ```bash
    cp .env.example .env
    ```
    Then, edit the `.env` file with your Supabase URL and Key.
5.  Run the backend server:
    ```bash
    uvicorn app.main:app --reload
    ```
The backend will be running at `http://localhost:8000`.