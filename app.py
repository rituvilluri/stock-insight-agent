from agent.agent_executor import agent_executor

if __name__ == "__main__":
    print("Welcome to the Stock Insight Agent!\n")
    while True:
        user_input = input("Ask a question (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent_executor.invoke(user_input)
        print("\nResponse:\n", response, "\n")
