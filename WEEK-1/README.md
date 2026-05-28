# 🤖 Project 1: Rule-Based AI Chatbot

## 📌 Overview

This project implements an **Advanced Rule-Based AI Chatbot** named **DecodeBot** using pure Python `if-else` logic. It simulates intelligent conversation by matching user inputs to predefined rules and responding accordingly — no external libraries required.

---

## 🎯 Goal

Build a conversational chatbot that can understand and respond to a variety of user inputs including greetings, questions, emotions, and educational queries using rule-based (heuristic) logic.

---

## 📁 Project Structure

```
Decodelabs Internship/
│
└── WEEK -1/
    ├── Chatbot.py       # Main chatbot code
    └── README.md        # Project documentation
```

---

## 🤖 Bot Details

| Property     | Details                          |
|--------------|----------------------------------|
| Bot Name     | DecodeBot                        |
| Type         | Rule-Based (Heuristic) Chatbot   |
| Language     | Python 3                         |
| Libraries    | None (pure Python)               |
| Interface    | Terminal / Command Line          |

---

## ⚙️ Tech Stack

| Tool       | Purpose                        |
|------------|--------------------------------|
| Python 3   | Programming language           |
| if-else    | Rule matching logic            |
| while loop | Keeps chatbot running          |
| input()    | Takes user input               |
| print()    | Displays bot responses         |

---

## 💬 Supported Commands

### Greetings
| User Input       | Bot Response                          |
|------------------|---------------------------------------|
| hello / hi / hey | Hello! Nice to meet you 😊            |
| good morning     | Good morning! Have a productive day ☀️ |
| good afternoon   | Good afternoon! 😊                    |
| good evening     | Good evening! Hope your day went well 🌙 |

### About the Bot
| User Input          | Bot Response                              |
|---------------------|-------------------------------------------|
| what is your name   | My name is DecodeBot 🤖                   |
| who created you     | I was created using Python and if-else logic |
| are you ai          | Yes! I am a simple rule-based AI chatbot  |

### Health & Feelings
| User Input   | Bot Response                          |
|--------------|---------------------------------------|
| how are you  | I am doing great! Thanks for asking 😊 |
| i am sad     | Don't worry. Better days are coming 💙 |
| i am happy   | That's wonderful to hear 😄            |
| i am tired   | Take some rest and stay hydrated 💧    |

### Education
| User Input              | Bot Response                               |
|-------------------------|--------------------------------------------|
| what is python          | Python is a popular programming language   |
| what is ai              | AI stands for Artificial Intelligence      |
| what is machine learning| Machine Learning allows systems to learn from data |
| what is chatbot         | A chatbot is a program that talks with users |

### Fun & General
| User Input      | Bot Response                                              |
|-----------------|-----------------------------------------------------------|
| tell me a joke  | Why do programmers prefer dark mode? Because light attracts bugs 😂 |
| tell me a fact  | Python was created by Guido van Rossum in 1991            |
| motivate me     | Success starts with consistency and hard work 💪           |
| sing a song     | La la la 🎵 ... I am not a great singer though 😅         |

### System Commands
| User Input  | Action                        |
|-------------|-------------------------------|
| help        | Shows all available commands  |
| bye / exit  | Stops the chatbot             |

---

## 🔁 How It Works

```
User types input
       ↓
Input converted to lowercase
       ↓
Matched against if-elif conditions
       ↓
Matching response printed by bot
       ↓
Loop continues until 'bye' or 'exit'
```

---

## 🚀 How to Run

### 1. Make sure Python is installed
```bash
python --version
```

### 2. Navigate to the project folder
```bash
cd "Decodelabs Internship/WEEK -1"
```

### 3. Run the chatbot
```bash
python Chatbot.py
```

### 4. Start chatting!
```
============================================
🤖 ADVANCED RULE-BASED AI CHATBOT
============================================
Type 'help' to see commands
Type 'bye' or 'exit' to stop the chatbot

You: hi
Bot: Hello! Nice to meet you 😊
You: motivate me
Bot: Success starts with consistency and hard work 💪
You: bye
Bot: Goodbye! Have a great day 👋
```

---

## 💡 Key Concepts Used

- **Rule-Based AI** — Responds based on predefined if-else conditions
- **Heuristic Logic** — Uses fixed rules instead of learning from data
- **while True Loop** — Keeps the chatbot alive until exit command
- **String Matching** — `.lower()` ensures case-insensitive matching
- **Break Statement** — Exits the loop cleanly on bye/exit.