"""
AI Therapist Application using Google Generative AI.

This script implements a conversational AI therapist that uses Google's
Generative AI API (e.g., Gemini). It loads a system prompt defining the AI's
personality, background information about the user from a separate file,
incorporates therapy notes from PDF files for context, and engages in a
therapeutic conversation with the user.

Prerequisites:
    - Python 3.10+
    - Google Generative AI API Key set as an environment variable: GEMINI_API_KEY
    - Path to therapy notes directory set as an environment variable: NOTES_DIR_PATH
    - Required libraries installed:
        pip install google-generativeai PyMuPDF pathlib

Setup:
    1. Create a file named `sysprompt.txt` in the same directory as this script
       containing the core AI personality and instructions (without user background).
    2. Create a file named `background.txt` in the same directory, containing
       the specific background information about the user.
    3. Set the `GEMINI_API_KEY` environment variable with your API key.
       Example (Linux/macOS): export GEMINI_API_KEY='YOUR_API_KEY'
       Example (Windows CMD): set GEMINI_API_KEY=YOUR_API_KEY
       Example (Windows PowerShell): $env:GEMINI_API_KEY='YOUR_API_KEY'
    4. Set the `NOTES_DIR_PATH` environment variable to the full path of the
       directory containing your PDF therapy notes.
       Example (Linux/macOS): export NOTES_DIR_PATH='/path/to/your/notes'
       Example (Windows CMD): set NOTES_DIR_PATH=C:\path\to\your\notes
       Example (Windows PowerShell): $env:NOTES_DIR_PATH='C:\path\to\your\notes'
    5. Ensure the directory specified by NOTES_DIR_PATH exists.

Execution:
    python your_script_name.py
"""

import os
from pathlib import Path

import fitz  # PyMuPDF
import google.generativeai as genai

# --- Constants ---
# Retrieve notes directory path from environment variable
NOTES_DIR_PATH_STR = os.environ.get("NOTES_DIR_PATH")
# Convert to Path object if the environment variable is set
NOTES_DIR = Path(NOTES_DIR_PATH_STR) if NOTES_DIR_PATH_STR else None

# System prompt file
SYS_PROMPT_FILE = Path("sysprompt.txt")
# User background information file
BACKGROUND_FILE = Path("background.txt")
# Google AI API Key (Loaded from environment variable)
API_KEY = os.environ.get("GEMINI_API_KEY")

# Define the model name as a constant
MODEL_NAME = "gemini-2.5-pro-exp-03-25"


# --- Function Definitions ---
def LoadTextFile(filePath: Path, fileDescription: str) -> str | None:
    """
    Loads text content from a specified file.

    Parameters
    ----------
    filePath : Path
        The path object pointing to the text file.
    fileDescription : str
        A description of the file being loaded (e.g., "system prompt",
        "user background") used for logging.

    Returns
    -------
    str | None
        The content of the file as a string, or None if the file cannot be read.

    """
    try:
        #
        fileContent = filePath.read_text(encoding="utf-8")
        #
        print(f"Successfully loaded {fileDescription} from: {filePath}")
        #
        return fileContent

    except FileNotFoundError:
        #
        print(f"Error: {fileDescription.capitalize()} file not found at {filePath}")
        #
        return None

    except IOError as e:
        #
        print(f"Error reading {fileDescription} file {filePath}: {e}")
        #
        return None


def LoadTherapyNotes(directory: Path | None) -> str:
    """
    Loads and extracts text from all PDF files in the specified directory.

    Parameters
    ----------
    directory : Path | None
        The path object pointing to the directory containing PDF notes,
        or None if the path was not provided or invalid.

    Returns
    -------
    str
        A single string containing the combined text extracted from all PDFs,
        separated by dividers. Returns an empty string if the directory is None,
        doesn't exist, no PDFs are found, or no text is extracted.

    """
    #
    if directory is None:
        #
        print(
            "Info: NOTES_DIR_PATH environment variable not set. "
            "Proceeding without therapy notes history."
        )
        #
        return ""

    #
    if not directory.is_dir():
        #
        print(
            f"Error: Therapy notes directory not found or is not a directory: {directory}. "
            "Proceeding without therapy notes history."
        )
        #
        return ""  # Return empty string if directory is invalid

    #
    notesTexts = []
    pdfFiles = list(directory.glob("*.pdf"))

    #
    if not pdfFiles:
        #
        print(f"Info: No PDF files found in {directory}.")
        #
        return ""  # Return empty string if no PDFs found

    #
    print(f"Found {len(pdfFiles)} PDF files in {directory}. Processing...")

    #
    for pdfPath in pdfFiles:
        #
        print(f"  Processing: {pdfPath.name}...")
        #
        try:
            #
            # Ensure file exists before opening
            if not pdfPath.is_file():
                print(f"  Warning: Skipping {pdfPath.name} as it's not a valid file.")
                continue

            #
            doc = fitz.open(pdfPath)
            #
            pdfText = ""
            #
            for pageNum, page in enumerate(doc.pages(), start=1):
                #
                pageText = page.get_text("text")
                if pageText:  # Only append if text was extracted
                    pdfText += pageText
                # Optional: Add a page break indicator for very long docs
                # if pageText: pdfText += f"\n--- Page {pageNum} End ---\n"
            #
            doc.close()
            #
            pdfTextStripped = pdfText.strip()
            if pdfTextStripped:
                #
                notesTexts.append(
                    f"--- Start of Notes from {pdfPath.name} ---\n{pdfTextStripped}\n"
                    f"--- End of Notes from {pdfPath.name} ---"
                )
                #
                print(f"  Successfully extracted text from {pdfPath.name}.")
            else:
                #
                print(f"  Warning: No text extracted from {pdfPath.name}.")

        except fitz.errors.FileDataError:  # More specific error for corrupt PDFs
            print(
                f"  Error: Could not process PDF file {pdfPath.name}. It might be corrupted or password-protected."
            )
        except Exception as e:
            #
            print(f"  Error processing PDF file {pdfPath.name}: {e}")
            # Continue to the next file

    #
    if not notesTexts:
        #
        print("Warning: Could not extract text from any valid PDF files found.")
        #
        return ""

    #
    # Combine all extracted notes into a single string
    combinedNotes = "\n\n".join(notesTexts)
    #
    print(f"Successfully combined therapy notes from {len(notesTexts)} PDF(s).")
    #
    return combinedNotes


def GenerateAiResponse(prompt: str, apiKey: str) -> str | None:
    """
    Sends the prompt to the Google AI API and returns the response.

    Handles API configuration and potential errors during generation.

    Parameters
    ----------
    prompt : str
        The complete prompt to send to the AI model, including system
        instructions, background, notes, and user input.
    apiKey : str
        The Google AI API key.

    Returns
    -------
    str | None
        The generated text response from the AI model, or None if an
        error occurred or the API key is missing.

    """
    #
    if not apiKey:
        #
        print("Error: GEMINI_API_KEY environment variable not set.")
        #
        return None

    #
    try:
        #
        # Configure the API client
        genai.configure(api_key=apiKey)

        #
        # Choose the model using the constant
        model = genai.GenerativeModel(MODEL_NAME)
        # Log which model is being used (consider removing in production)
        # print(f"Using model: {MODEL_NAME}")

        #
        # --- Safety Settings (Example - Adjust as needed) ---
        # Block potentially harmful content. You might adjust thresholds.
        # Refer to Google AI documentation for details on categories and thresholds.
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE",
            },
        ]

        #
        # Generate content
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # --- Process Response ---
        # Check for blocked prompt or response
        if not response.parts:
            #
            block_reason = "Unknown"
            if (
                hasattr(response, "prompt_feedback")
                and response.prompt_feedback.block_reason
            ):
                block_reason = response.prompt_feedback.block_reason
            #
            print(
                f"\nWarning: AI response was empty or blocked. Reason: {block_reason}"
            )
            # Provide a safe default response
            return (
                "I apologize, but I encountered an issue or the content was blocked "
                "by safety filters. Could you please rephrase or try a different topic?"
            )

        #
        # Return the generated text
        return response.text

    # Handle specific API errors if possible, e.g., AuthenticationError, ResourceExhaustedError
    except google.api_core.exceptions.PermissionDenied as e:
        print(
            f"\nAPI Error: Permission Denied. Check if the API key is valid, has access to "
            f"the model '{MODEL_NAME}', and the Gemini API is enabled in your Google Cloud project. Details: {e}"
        )
        return None
    except google.api_core.exceptions.NotFound as e:
        print(
            f"\nAPI Error: Model Not Found. The model name '{MODEL_NAME}' might be incorrect "
            f"or unavailable. Details: {e}"
        )
        return None
    except Exception as e:
        #
        # Catch other potential API errors
        print(f"\nError during AI response generation: {e}")
        print(
            "Check API key, internet connection, model name, quota limits, and prompt content."
        )
        #
        return None


def main():
    """
    Main function to run the AI Therapist application.

    Loads prompts, background, notes, handles the conversation loop, and interacts with the AI API.
    """
    print("--- AI Therapist Initializing ---")

    # --- Validate Environment Variables ---
    #
    if not NOTES_DIR:
        print("Warning: 'NOTES_DIR_PATH' environment variable not set or invalid.")
        # Continue without notes, LoadTherapyNotes will handle the None value.
    #
    if not API_KEY:
        #
        print("\nFatal Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set the environment variable and restart the application.")
        #
        return

    # --- Load Core AI Prompt ---
    #
    baseSystemPrompt = LoadTextFile(SYS_PROMPT_FILE, "system prompt")
    #
    if baseSystemPrompt is None:
        #
        print("Fatal Error: Could not load system prompt. Exiting.")
        #
        return  # Exit if system prompt is essential

    # --- Load User Background ---
    #
    userBackground = LoadTextFile(BACKGROUND_FILE, "user background")
    #
    if userBackground is None:
        #
        print("Warning: Could not load user background file. Proceeding without it.")
        userBackground = (
            "No user background information was loaded."  # Provide default fallback
        )

    # --- Load Therapy Notes ---
    #
    therapyNotesText = LoadTherapyNotes(NOTES_DIR)
    # therapyNotesText will be an empty string if dir invalid, no PDFs, or no text extracted

    # --- Construct the Full System Context for the AI ---
    #
    # Start with the core AI instructions
    fullSystemContext = baseSystemPrompt

    #
    # Append the user background information
    fullSystemContext += (
        "\n\n"
        "--- USER BACKGROUND INFORMATION ---\n"
        "This section contains important background information about the user you are "
        "interacting with. Use this information to personalize your responses and "
        "understand their context.\n\n"
        f"{userBackground}\n\n"
        "--- END USER BACKGROUND INFORMATION ---"
    )

    #
    # Append therapy notes history, if available
    if therapyNotesText:
        #
        fullSystemContext += (
            "\n\n"
            "--- BEGIN THERAPY NOTES HISTORY ---\n"
            "The following are notes from the user's previous therapy sessions. Use this "
            "history in conjunction with the background information to inform your "
            "responses. Therapy notes reflect recent developments. If there are contradictions "
            "between the general background and the therapy notes, prioritize the "
            "information in the therapy notes as they are more current.\n\n"
            f"{therapyNotesText}\n\n"
            "--- END THERAPY NOTES HISTORY ---"
        )
    else:
        #
        fullSystemContext += (
            "\n\n"
            "--- THERAPY NOTES HISTORY ---\n"
            "No therapy notes were loaded or found. Base your responses primarily "
            "on the user background information provided above and the ongoing "
            "conversation."
            "\n--- END THERAPY NOTES HISTORY ---"
        )

    # --- Add Ethical Guidelines Reminder ---
    #
    fullSystemContext += (
        "\n\n--- IMPORTANT REMINDERS ---\n"
        "1. Role: You are an AI assistant for supportive conversation, NOT a licensed "
        "therapist or medical professional. Do not provide diagnoses or treatment plans.\n"
        "2. Ethics: Adhere strictly to the ethical guidelines and personality defined in "
        "your initial system prompt.\n"
        "3. Crisis: If the user mentions self-harm, suicide, or abuse, respond empathetically, "
        "urge them to seek immediate professional help, and provide crisis resources "
        "(e.g., 988 Suicide & Crisis Lifeline in the US). Do not attempt to manage a crisis "
        "yourself.\n"
        "4. Context: Continuously refer to the User Background and Therapy Notes provided "
        "to maintain relevant and personalized conversation.\n"
        "--- END IMPORTANT REMINDERS ---"
    )

    # --- Start Conversation Loop ---
    #
    print("\n--- AI Therapist Ready ---")
    print(f"Model: {MODEL_NAME}")
    print("Type 'quit' or 'exit' to end the session.")
    print(
        "Disclaimer: This AI is for supportive conversation and is not a substitute "
        "for professional therapy or medical advice."
    )

    #
    # Initialize conversation history (optional, but good for context window management if needed later)
    # conversation_history = [] # Example: list of {'role': 'user'/'model', 'parts': [text]}

    #
    try:
        #
        while True:
            #
            # Get user input
            try:
                #
                userInput = input("\nYou: ")
            except EOFError:  # Handle Ctrl+D
                #
                print("\nAI Therapist: Ending session due to input closure. Take care.")
                #
                break

            #
            # Check for exit command
            if userInput.lower() in ["quit", "exit"]:
                #
                print("\nAI Therapist: Ending session. Take care.")
                #
                break

            #
            # Construct the full prompt for this turn
            # This includes the static context plus the latest user message
            # For very long conversations, context window management might be needed
            # (e.g., summarizing earlier parts), but for now, send the full context.
            currentPrompt = f"{fullSystemContext}\n\nUser: {userInput}\n\nAI Therapist:"

            # --- Generate AI Response ---
            #
            print("\nAI Therapist: (Thinking...)")  # Thinking indicator
            #
            aiResponse = GenerateAiResponse(currentPrompt, API_KEY)

            #
            # Print the response or error message
            if aiResponse:
                #
                print(f"\nAI Therapist: {aiResponse}")
                # Optional: Add user input and AI response to history
                # conversation_history.append({'role': 'user', 'parts': [userInput]})
                # conversation_history.append({'role': 'model', 'parts': [aiResponse]})
            else:
                #
                # Error messages are printed within GenerateAiResponse
                print(
                    "\nAI Therapist: I encountered an issue generating a response. "
                    "Please check the console logs for details and try again."
                )
                # Consider adding a retry mechanism or specific guidance based on error type

    except KeyboardInterrupt:
        #
        print("\n\nAI Therapist: Session interrupted by user. Goodbye.")
    except Exception as e:
        #
        print(f"\nAn unexpected error occurred in the main loop: {e}")
    finally:
        #
        print("\n--- AI Therapist Session Ended ---")


# --- Main Execution ---
#
if __name__ == "__main__":
    #
    main()
