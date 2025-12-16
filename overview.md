# FLEX-Med

FLEX-Med is a comprehensive platform for medical image analysis, incorporating a machine learning backend, a web-based frontend, and a federated learning component for distributed model training.

## Components

The project is divided into three main components:

### 1. Backend

The backend is a [FastAPI](https://fastapi.tiangolo.com/) application that serves a machine learning model for medical image prediction.

**Features:**

-   **Prediction API:** Endpoints for uploading medical images and receiving predictions.
-   **PyTorch Integration:** Uses a PyTorch model for inference.
-   **Health Check:** An endpoint to monitor the status of the API.

**To get started with the backend, refer to the detailed instructions in the [backend/README.md](backend/README.md).**

### 2. Frontend

The frontend is a [Next.js](https://nextjs.org/) web application that provides a user-friendly interface for interacting with the FLEX-Med platform.

**Features:**

-   **User Authentication:** Secure user authentication using [Supabase](https://supabase.io/).
-   **Interactive UI:** A rich user interface built with [shadcn/ui](https://ui.shadcn.com/) and [Tailwind CSS](https://tailwindcss.com/).
-   **Backend Integration:** Seamlessly communicates with the backend API to provide predictions to the user.

**To get started with the frontend, refer to the detailed instructions in the [frontend/README.md](frontend/README.md).**

### 3. Federated Learning

The federated learning component uses the [Flower](https://flower.ai/) framework to train machine learning models in a distributed and privacy-preserving manner.

**Features:**

-   **Federated Training:** Enables training models on decentralized data without compromising patient privacy.
-   **PyTorch and Flower:** Built with PyTorch and the Flower framework.
-   **Simulation and Deployment:** Supports both simulated and real-world federated learning deployments.

**To get started with the federated learning component, refer to the detailed instructions in the [federated_learning/README.md](federated_learning/README.md).**

## Technologies Used

-   **Backend:** FastAPI, Python, PyTorch
-   **Frontend:** Next.js, React, TypeScript, Supabase, shadcn/ui, Tailwind CSS
-   **Federated Learning:** Flower, Python, PyTorch

## Overview

The FLEX-Med platform is designed to be a flexible and scalable solution for medical image analysis. The federated learning component allows for the continuous improvement of the machine learning model without centralizing sensitive patient data. The backend provides a robust API for serving the trained model, and the frontend offers an intuitive interface for healthcare professionals to use the platform.
