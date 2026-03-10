# Final Project Name

> Short one-line description of what the project does.

A brief 2-3 sentence description of the project, its purpose, and the main
technologies used. Mention the problem it solves and who it is for.

---

## Authors

- Ayaa Asoba
- Pablo González Martín 
- Xavier Bruneau 


---

## Requirements

- Python **3.10+**


Check your Python version before starting:

```bash
python --version
# or
python3 --version
```

---

## Getting Started

### 1 — Clone the repository

Move to the folder you want to clone the repository in.

```bash
git clone 
cd your-repo
```

### 2 — Create and activate virtual environment

```bash
# Create
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```


## Project Structure

```
your-repo/
├── data/                   # Raw and processed data (gitignored)
│   ├── raw/
│   └── processed/
├── models/                 # Saved model checkpoints (gitignored)
├── notebooks/              # Jupyter exploration notebooks
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model definitions
│   ├── training/           # Training loops and evaluation
│   └── utils/              # Shared helpers
├── .gitignore           
├── requirements.txt
└── README.md
```
