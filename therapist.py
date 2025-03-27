# -*- coding: utf-8 -*-
"""
AI Therapist Application using Google Generative AI (Multi-Turn Chat).

This script implements a conversational AI therapist that uses Google's
Generative AI API (e.g., Gemini) and maintains conversation history within
a single session. It loads a system prompt, user background, therapy notes,
and engages in a therapeutic conversation.

Prerequisites:
    - Python 3.10+
    - Google Generative AI API Key set as an environment variable: GEMINI_API_KEY
    - Path to therapy notes directory set as an environment variable: NOTES_DIR_PATH
    - Required libraries installed:
        pip install google-generativeai PyMuPDF python-dotenv pathlib natsort

Setup:
    1. Create `sysprompt.txt` (core AI instructions).
    2. Create `background.txt` (user background info).
    3. Create `.env` file (optional) or set environment variables:
       GEMINI_API_KEY='YOUR_API_KEY'
       NOTES_DIR_PATH='/full/path/to/your/notes'
    4. Ensure the notes directory exists.

Execution:
    python your_script_name.py
"""

import os
from pathlib import Path

import fitz  # PyMuPDF
import google.api_core.exceptions
import google.generativeai as genai
from dotenv import load_dotenv
from natsort import natsorted  # Import natsorted

# Load environment variables from .env file, if it exists
load_dotenv()

# --- Constants ---

NOTES_DIR_PATH_STR = os.environ.get("NOTES_DIR_PATH")
NOTES_DIR = Path(NOTES_DIR_PATH_STR) if NOTES_DIR_PATH_STR else None
SYS_PROMPT_FILE = Path("sysprompt.txt")
BACKGROUND_FILE = Path("background.txt")
API_KEY = os.environ.get("GEMINI_API_KEY")

# Warning: 'gemini-2.5-pro-exp-03-25' might be experimental. Use 'gemini-pro' or 'gemini-1.5-pro-latest' if issues arise.
MODEL_NAME = "gemini-2.5-pro-exp-03-25"

# --- Safety Settings (Example - Adjust as needed) ---
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

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
        print("Info: NOTES_DIR_PATH env var not set. No therapy notes loaded.")

        #
        return ""

    #
    if not directory.is_dir():
        #
        print(
            f"Error: Notes directory not found/invalid: {directory}. No notes loaded."
        )

        #
        return ""

    #
    notesTexts = []
    # Use natsorted for natural sorting (e.g., file2.pdf before file10.pdf)
    pdfFiles = natsorted(list(directory.glob("*.pdf")))

    #
    if not pdfFiles:
        #
        print(f"Info: No PDF files found in {directory}.")

        #
        return ""

    #
    print(f"Found {len(pdfFiles)} PDF files in {directory}. Processing...")
    #
    processed_count = 0

    #
    for pdfPath in pdfFiles:
        #
        print(f"  Processing: {pdfPath.name}...")

        #
        try:
            #
            if not pdfPath.is_file():
                #
                print(f"  Warning: Skipping {pdfPath.name}, not a valid file.")
                continue

            #
            doc = fitz.open(pdfPath)
            pdfText = "".join(
                page.get_text("text") for page in doc.pages() if page.get_text("text")
            )
            doc.close()

            #
            pdfTextStripped = pdfText.strip()

            #
            if pdfTextStripped:
                #
                notesTexts.append(
                    f"--- Start Notes: {pdfPath.name} ---\n{pdfTextStripped}\n--- End Notes: {pdfPath.name} ---"
                )
                print(f"  Successfully extracted text from {pdfPath.name}.")
                processed_count += 1
            #
            else:
                #
                print(f"  Warning: No text extracted from {pdfPath.name}.")

        except fitz.errors.FileDataError:
            #
            print(f"  Error: Corrupt/password-protected PDF: {pdfPath.name}.")
        #
        except Exception as e:
            #
            print(f"  Error processing PDF {pdfPath.name}: {e}")

    #
    if not notesTexts:
        #
        print("Warning: Could not extract text from any valid PDF files.")

        #
        return ""

    #
    print(f"Successfully combined therapy notes from {processed_count} PDF(s).")

    #
    # Combine all extracted notes into a single string
    combinedNotes = "\n\n".join(notesTexts)

    #
    return combinedNotes


def main():
    """
    Main function to run the AI Therapist application with multi-turn chat.
    """

    print("--- AI Therapist Initializing ---")

    # --- Validate Environment Variables ---

    #
    if not API_KEY:
        #
        print("\nFatal Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the environment variable (or use a .env file) and restart.")

        #
        return

    #
    if not NOTES_DIR_PATH_STR:  # Check if the path string was retrieved
        #
        print("Warning: 'NOTES_DIR_PATH' environment variable not set or invalid.")
        # NOTES_DIR will be None, LoadTherapyNotes handles this.

    # --- Load Static Context ---

    #
    baseSystemPrompt = LoadTextFile(SYS_PROMPT_FILE, "system prompt")

    #
    if baseSystemPrompt is None:
        #
        print("Fatal Error: Could not load system prompt. Exiting.")

        #
        return

    #
    userBackground = LoadTextFile(BACKGROUND_FILE, "user background")

    #
    if userBackground is None:
        #
        print("Warning: Could not load user background file. Using fallback.")
        userBackground = "No user background information was loaded."

    #
    therapyNotesText = LoadTherapyNotes(NOTES_DIR)  # Handles None NOTES_DIR

    # --- Construct the SINGLE System Instruction ---
    # Combine all static context into one block for the model's system instruction

    #
    system_instruction_parts = [baseSystemPrompt]

    #
    system_instruction_parts.append(
        "\n\n--- USER BACKGROUND INFORMATION ---\n"
        "Use this information to personalize responses and understand context.\n\n"
        f"{userBackground}\n\n"
        "--- END USER BACKGROUND INFORMATION ---"
    )

    #
    if therapyNotesText:
        #
        system_instruction_parts.append(
            "\n\n--- BEGIN THERAPY NOTES HISTORY ---\n"
            "Use these notes (prioritizing over background if conflicting) for context.\n\n"
            f"{therapyNotesText}\n\n"
            "--- END THERAPY NOTES HISTORY ---"
        )
    #
    else:
        #
        system_instruction_parts.append(
            "\n\n--- THERAPY NOTES HISTORY ---\n"
            "No therapy notes loaded. Base responses on background and conversation."
            "\n--- END THERAPY NOTES HISTORY ---"
        )

    #
    # Append Ethical Guidelines Reminder
    system_instruction_parts.append(
        "\n\n--- IMPORTANT REMINDERS ---\n"
        "1. Role: AI assistant, NOT licensed therapist. No diagnoses/treatment.\n"
        "2. Ethics: Adhere to initial system prompt guidelines.\n"
        "3. Crisis: If user mentions self-harm/suicide/abuse, respond empathetically, "
        "urge professional help, provide crisis resources (e.g., 988 US). Do not manage crisis.\n"
        "4. Context: Refer to User Background/Therapy Notes.\n"
        "--- END IMPORTANT REMINDERS ---"
    )

    #
    combined_system_instruction = "".join(system_instruction_parts)

    # --- Initialize Google AI ---

    #
    try:
        #
        genai.configure(api_key=API_KEY)

        #
        # Create the model with the combined system instruction
        model = genai.GenerativeModel(
            MODEL_NAME,
            safety_settings=SAFETY_SETTINGS,
            system_instruction=combined_system_instruction,
        )

        #
        # Start a chat session (history is managed internally)
        chat = model.start_chat(history=[])  # Start with empty user/model turn history

        #
        print("\n--- AI Therapist Ready ---")
        print(f"Model: {MODEL_NAME}")
        print("Type 'quit' or 'exit' to end the session.")
        print("Disclaimer: AI for support, not professional therapy/medical advice.")

    except Exception as e:
        #
        print(f"\nFatal Error during AI Initialization: {e}")
        print("Check API key, model name, and network connection.")

        #
        return

    # --- Conversation Loop ---

    #
    try:
        #
        while True:
            #
            # Get user input
            try:
                #
                userInput = input("\nYou: ").strip()

                #
                if not userInput:  # Handle empty input
                    continue

            except EOFError:
                #
                print("\nAI Therapist: Input stream closed. Ending session.")

                #
                break

            #
            # Check for exit command
            if userInput.lower() in ["quit", "exit"]:
                #
                print("\nAI Therapist: Ending session as requested. Take care.")

                #
                break

            # --- Send Message and Get Response ---

            #
            print("\nAI Therapist: (Thinking...)")

            #
            try:
                #
                # Send message - chat history is automatically updated by the object
                response = chat.send_message(userInput)

                # --- Process Response ---
                # Check for blocked content or other issues

                #
                if not response.parts:
                    #
                    block_reason = "Unknown"

                    #
                    if (
                        hasattr(response, "prompt_feedback")
                        and response.prompt_feedback.block_reason
                    ):
                        #
                        block_reason = (
                            response.prompt_feedback.block_reason.name
                        )  # Use .name for enum
                    #
                    elif (
                        hasattr(response, "candidates")
                        and response.candidates
                        and response.candidates[0].finish_reason.name != "STOP"
                    ):
                        #
                        block_reason = f"Finish Reason: {response.candidates[0].finish_reason.name}"

                    #
                    print(
                        f"\nWarning: AI response empty/blocked. Reason: {block_reason}"
                    )
                    aiResponseText = (
                        "I apologize, but I encountered an issue or the content was blocked. "
                        "Could you rephrase or try a different topic?"
                    )
                #
                else:
                    #
                    aiResponseText = response.text

                #
                # Print the AI's response
                print(f"\nAI Therapist: {aiResponseText}")

            # --- API Error Handling within the loop ---
            except google.api_core.exceptions.PermissionDenied as e:
                #
                print(
                    f"\nAPI Error: Permission Denied. Check API key/model access. Details: {e}"
                )
                # Decide whether to break or allow retry
                # break
            #
            except google.api_core.exceptions.NotFound as e:
                #
                print(
                    f"\nAPI Error: Model '{MODEL_NAME}' Not Found or unavailable. Details: {e}"
                )

                #
                break  # Likely fatal for this session
            #
            except google.api_core.exceptions.ResourceExhausted as e:
                #
                print(f"\nAPI Error: Quota Exceeded. Details: {e}")
                # Maybe wait and retry, or break

                #
                break
            #
            except google.api_core.exceptions.InvalidArgument as e:
                #
                print(
                    f"\nAPI Error: Invalid Argument (check prompt/content). Details: {e}"
                )
                # Log problematic input if needed (beware sensitive data)
                # print(f"Input causing error: {userInput}")
            #
            except Exception as e:
                #
                print(f"\nError during AI response generation: {e}")
                # General error, maybe allow user to try again

    # --- General Error Handling & Cleanup ---
    except KeyboardInterrupt:
        #
        print("\n\nAI Therapist: Session interrupted by user. Goodbye.")
    #
    except Exception as e:
        #
        print(f"\nAn unexpected error occurred in the main loop: {e}")
    #
    finally:
        #
        print("\n--- AI Therapist Session Ended ---")


# --- Main Execution ---

#
if __name__ == "__main__":
    #
    main()
