from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_cohere import ChatCohere
from pydantic import BaseModel

class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str


DEFAULT_QUESTION = "What's the capital of Mars?"


def get_user_question() -> str:
    """Prompt user for a question with a default value."""
    try:
        return input(f"Question [{DEFAULT_QUESTION}]: ").strip() or DEFAULT_QUESTION
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting...")
        exit(0)

def handle_error(e: Exception, context: str = "") -> None:
    """Handle and log errors."""
    print(f"\nError {context}: {e}")
    if hasattr(e, 'response') and hasattr(e.response, 'text'):
        print(f"Response: {e.response.text}")

def main():
    """Main execution function."""
    load_dotenv()
    model = ChatCohere()
    
    # Use structured output directly with the model
    structured_model = model.with_structured_output(CapitalInfo)
    
    question = get_user_question()
    print(f"\nProcessing: {question}\n")
    
    # Enhanced prompt for creative responses
    system_prompt = "You are a science fiction writer. Be creative and detailed in your response."
    
    try:
        result = structured_model.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ])
        print(f"Name: {result.name}")
        print(f"Location: {result.location}")
        print(f"Vibe: {result.vibe}")
        print(f"Economy: {result.economy}")
    except Exception as e:
        handle_error(e, "in main execution")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        exit(0)
    except Exception as e:
        handle_error(e, "in main execution")
