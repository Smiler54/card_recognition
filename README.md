# Poker PLO5 Card Recognition & Copy to Clipboard

A basic OpenCV project that captures poker game window from your screen and detects your cards and recognize them in real-time.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

- Click [Start Capture] button.
- The program will start to capture the Poker Game Window in real time.
- And it will seperate multiple tables, detect each cards and recognize them. (such as "Kh 10h 10d 9c 3s")

## Features

- Real-time window capture
- Seperate multiple tables
- Match templates
