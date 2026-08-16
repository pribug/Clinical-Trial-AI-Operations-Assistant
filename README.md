# Clinical Trial AI Operations Assistant

An AI-assisted clinical research operations tool designed to support preliminary study planning and research-site feasibility assessment.

The application combines a local Large Language Model (LLM) with deterministic Python-based validation and scoring. The goal is to demonstrate how AI can assist clinical research workflows while keeping structured checks, numerical calculations, and human review outside the LLM.

> **Important:** This is a research/portfolio demonstration tool. It is not a clinical decision-support system and does not provide clinical, statistical, regulatory, or scientific validation.

---

## Overview

Clinical research workflows often involve unstructured study requirements, feasibility assessment, and operational communication.

This project provides a lightweight interface for two research operations workflows:

1. **Preliminary Study Planning**
2. **Research Site Feasibility**

The application uses AI where language generation is useful, while deterministic Python logic is used for structured verification and numerical scoring.

---

## Key Features

### 1. Preliminary Study Specification

Users can enter a natural-language study idea and generate a structured preliminary study specification.

The system extracts information such as:

- Study title
- Objective
- Study design
- Population
- Primary endpoint
- Target sample size
- Study duration
- Key visits
- Assumptions

The model is instructed **not to invent missing clinical-study details**.

When information is not provided, the system identifies it as requiring human review rather than filling the gap with unsupported assumptions.

---

### 2. Automated Protocol Verification

The generated study specification can be checked using deterministic Python rules.

The verification layer identifies issues such as:

- Missing study design
- Missing population information
- Missing endpoint information
- Missing study duration
- Missing sample-size information
- AI-inferred information
- Unsupported assumptions
- Missing visit schedules
- Statistical review requirements

The system distinguishes between:

- **Missing information**
- **AI-inferred information**
- **Statistical review**
- **Human review**

The LLM is used only to explain the results of these structured checks. It does not perform the verification itself.

---

### 3. Research Site Feasibility

The application provides a transparent weighted scoring model for comparing research sites.

Users can either:

- Use the built-in demonstration dataset, or
- Upload an Excel file containing site-feasibility data.

The scoring model considers:

| Factor | Weight |
|---|---:|
| Monthly Patients | 40% |
| Enrollment Speed | 30% |
| EDC Experience | 20% |
| Active Trial Load | 10% |

Higher patient availability and faster enrollment receive higher scores.

Sites with EDC experience receive a positive score, while active trial load is treated as a competing operational factor.

The final score is calculated programmatically using Python.

---

### 4. Target Enrollment Feasibility

The user can specify a target number of participants.

The application uses the target enrollment requirement to calculate:

- Estimated months required to reach the target
- Target-related feasibility contribution
- Updated site feasibility scores

This allows users to explore how different enrollment requirements can affect site feasibility.

---

### 5. AI Feasibility Explanation

After Python calculates the site ranking, the LLM provides a concise explanation of:

- Why higher-ranked sites performed well
- Important trade-offs
- Potential sites for further feasibility review
- Factors requiring human confirmation

The LLM does **not** calculate or modify the numerical scores.

---

### 6. Site Feasibility Email Generator

The application can generate a concise feasibility inquiry for a selected research site.

The draft can request confirmation of:

- Participant enrollment capability
- Expected enrollment timeline

The generated email is explicitly presented as a draft requiring human review before sending.

---

### 7. Audit Logging

Key Study Setup actions are recorded in an audit log.

Logged activities include:

- Protocol generation
- Protocol verification
- Site feasibility ranking
- Site communication generation

The audit log can be downloaded as a CSV file for record keeping.

---

## Architecture

The application follows a hybrid AI + deterministic architecture:

```text
                         User
                           |
                           v
                  Streamlit Interface
                           |
              +------------+------------+
              |                         |
              v                         v
       Study Planning          Site Feasibility
              |                         |
              v                         v
         Local LLM                Python Scoring
              |                         |
              v                         v
     Structured Output          Numerical Results


              |                         |
              v                         v
    Python Verification        LLM Explanation
              |                         |
              +------------+------------+
                           |
                           v
                     Human Review
```

# Design Principle

The project intentionally avoids giving the LLM responsibility for tasks that can be handled deterministically.

LLM Responsibilities
Natural-language study specification generation
Concise explanation of verification findings
Explanation of site-ranking results
Feasibility email drafting
Python Responsibilities
Structured validation
Missing-information detection
Sample-size review flags
Site feasibility calculations
Target-enrollment calculations
Ranking
Audit logging

This separation improves transparency and makes the numerical components reproducible.

Technology Stack
Python
Streamlit
Ollama
Llama 3.2 3B
LangChain
Pandas
Regular Expressions
Excel / CSV processing
Local AI

The application uses Ollama to run the language model locally.

Current Model
llama3.2:3b

No external LLM API is required for the current implementation.

# Installation
1. Clone the repository
git clone <your-repository-url>
cd Clinical_Trial_Op_AI_Assistant
2. Create a virtual environment
python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Install Ollama

Install Ollama for your operating system and verify that it is available from the terminal.

Check:

ollama --version
5. Pull the model
ollama pull llama3.2:3b

Verify:

ollama list

You should see:

llama3.2:3b
6. Run the application
streamlit run app.py

The application will open in your browser.

# Example Workflow
Protocol Generator

Example input:

Conduct a randomized double-blind study comparing Treatment A
with placebo in 100 adults with psoriasis. The primary endpoint
is PASI improvement at Week 12.

The system can extract the explicitly provided information while leaving unspecified elements for human review.

For example:

Study Design

Randomized double-blind

Population

100 adults with psoriasis

Primary Endpoint

PASI improvement at Week 12

Target Sample Size

100

Study Duration

Not specified - requires human review

The system does not attempt to statistically justify the target sample size.

Site Feasibility

The demonstration dataset contains multiple research sites with operational variables such as:

Monthly patient availability
Active trial load
EDC experience
Average enrollment time

Users can enter a target enrollment requirement and compare the resulting feasibility estimates and rankings.

# Disclaimer

This project is intended for educational, research, and portfolio demonstration purposes only.

It does not replace qualified clinical researchers, statisticians, investigators, regulatory professionals, or other subject-matter experts.

No output generated by the application should be used as a substitute for professional clinical, statistical, scientific, regulatory, or operational review.
