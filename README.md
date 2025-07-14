# TraeGPT

A project powered by AI for advanced code assistance and search.

## Features
- AI-powered code suggestions
- Semantic and regex code search
- Customizable workflows

## Getting Started
1. **Install Python**
   - Download and install Python (3.11 is recommended for this project) from the [official website](https://www.python.org/downloads/).
   - Make sure to add Python to your system PATH during installation.

2. **Install Ollama**
   - Visit the [Ollama installation page](https://ollama.com/download) and follow the instructions for your operating system.
   - Pull the AI model that is being used; using the `ollama pull HammerAI/mistral-nemo-uncensored` command in your terminal (CMD)
   - For more details, see the [Ollama documentation](https://github.com/jmorganca/ollama#installation).

3. **Clone the repository:**
   ```bash
   git clone https://github.com/zabwie/TraeGPT.git
   ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up and run the bot:**
   - Make sure Ollama is running on your machine.
   - Run the main application script
     ```bash
     python ollama.py
     ```
   - The bot should now be active and ready to assist you.