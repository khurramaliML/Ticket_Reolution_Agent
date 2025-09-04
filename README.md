
# Ticket Resolution Agent

## Overview
Ticket Resolution Agent is an advanced Python-based solution designed to automate and streamline the process of resolving customer support tickets. Leveraging state-of-the-art language models, it classifies tickets, retrieves relevant context, drafts professional responses, and ensures quality through automated review cycles.

## Features
- **Automated Ticket Classification:** Uses LLMs to categorize tickets for efficient routing.
- **Contextual Knowledge Retrieval:** Fetches relevant information based on ticket type.
- **Response Drafting:** Generates empathetic, actionable, and professional replies.
- **Quality Assurance:** Reviews responses for accuracy and compliance with support guidelines.
- **Retry and Refinement:** Iteratively improves responses based on feedback.

## Setup Instructions

### 1. Clone the Repository
```powershell
git clone https://github.com/khurramaliML/Ticket_Reolution_Agent.git
cd Ticket_Reolution_Agent
```

### 2. Create and Activate a Python Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
Install required packages (ensure you have a `requirements.txt`):
```powershell
pip install -r requirement.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root and add your API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the Agent
```powershell
python Ticket_Reolution_Agent/agent.py
```

## Usage
The agent will process a sample ticket and output the classification, context, draft response, review feedback, and the final response. You can customize the initial ticket details in `agent.py`.
