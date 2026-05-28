print("=" * 50)
print("🤖 ADVANCED RULE-BASED AI CHATBOT")
print("=" * 50)
print("Type 'help' to see commands")
print("Type 'bye' or 'exit' to stop the chatbot\n")

while True:

    user_input = input("You: ").lower()

    # Greetings
    if user_input == "hello" or user_input == "hi" or user_input == "hey":
        print("Bot: Hello! Nice to meet you 😊")

    elif user_input == "good morning":
        print("Bot: Good morning! Have a productive day ☀️")

    elif user_input == "good afternoon":
        print("Bot: Good afternoon! 😊")

    elif user_input == "good evening":
        print("Bot: Good evening! Hope your day went well 🌙")

    # Asking about bot
    elif user_input == "what is your name":
        print("Bot: My name is DecodeBot 🤖")

    elif user_input == "who created you":
        print("Bot: I was created using Python and if-else logic.")

    elif user_input == "are you ai":
        print("Bot: Yes! I am a simple rule-based AI chatbot.")

    # Health & feelings
    elif user_input == "how are you":
        print("Bot: I am doing great! Thanks for asking 😊")

    elif user_input == "i am sad":
        print("Bot: Don't worry. Better days are coming 💙")

    elif user_input == "i am happy":
        print("Bot: That's wonderful to hear 😄")

    elif user_input == "i am tired":
        print("Bot: Take some rest and stay hydrated 💧")

    # Basic conversation
    elif user_input == "what are you doing":
        print("Bot: I am chatting with you!")

    elif user_input == "tell me a joke":
        print("Bot: Why do programmers prefer dark mode? Because light attracts bugs 😂")

    elif user_input == "tell me a fact":
        print("Bot: Python was created by Guido van Rossum in 1991.")

    elif user_input == "motivate me":
        print("Bot: Success starts with consistency and hard work 💪")

    # Education related
    elif user_input == "what is python":
        print("Bot: Python is a popular programming language.")

    elif user_input == "what is ai":
        print("Bot: AI stands for Artificial Intelligence.")

    elif user_input == "what is machine learning":
        print("Bot: Machine Learning allows systems to learn from data.")

    elif user_input == "what is chatbot":
        print("Bot: A chatbot is a program that talks with users.")

    # Personal questions
    elif user_input == "my name is harsh":
        print("Bot: Nice to meet you, Harsh!")

    elif user_input == "do you like humans":
        print("Bot: Of course! Humans created me 😄")

    elif user_input == "can you help me":
        print("Bot: Yes! I will try my best to help you.")

    # Time pass responses
    elif user_input == "bored":
        print("Bot: Try learning something new today!")

    elif user_input == "sing a song":
        print("Bot: La la la 🎵 ... I am not a great singer though 😅")

    elif user_input == "do you sleep":
        print("Bot: No, I work 24/7 🤖")

    # Help command
    elif user_input == "help":
        print("\nAvailable commands:")
        print("- hello / hi / hey")
        print("- how are you")
        print("- what is your name")
        print("- tell me a joke")
        print("- tell me a fact")
        print("- what is python")
        print("- what is ai")
        print("- motivate me")
        print("- bye / exit\n")

    # Exit conditions
    elif user_input == "bye" or user_input == "exit":
        print("Bot: Goodbye! Have a great day 👋")
        break

    # Empty input
    elif user_input == "":
        print("Bot: Please type something.")

    # Default response
    else:
        print("Bot: Sorry, I don't understand that yet.")