# -*- coding: utf-8 -*-
"""
AI Therapist Application using Google Generative AI (Multi-Turn Chat).

This script implements a conversational AI therapist that uses Google's
Generative AI API (e.g., Gemini) and maintains conversation history within
a single session. It loads a system prompt, user background, therapy notes,
implements rate limiting, logs chat sessions, ensures .gitignore rules,
and engages in a therapeutic conversation with enhanced console output using Rich.

Prerequisites:
    - Python 3.10+
    - Google Generative AI API Key set as an environment variable: GEMINI_API_KEY
    - Path to therapy notes directory set as an environment variable: NOTES_DIR_PATH
    - Required libraries installed:
        pip install google-generativeai PyMuPDF python-dotenv pathlib natsort rich

Setup:
    1. Create `sysprompt.txt` (core AI instructions).
    2. Create `background.txt` (user background info).
    3. Create `.env` file (optional) or set environment variables:
       GEMINI_API_KEY='YOUR_API_KEY'
       NOTES_DIR_PATH='/full/path/to/your/notes'
    4. Ensure the notes directory exists.
    5. A 'history' directory will be created automatically for chat logs.
    6. A '.gitignore' file will be checked/created to ignore sensitive files.

Execution:
    python your_script_name.py
"""

import datetime  # For logging filenames
import os
import time  # For rate limiting and logging
from collections import deque  # For rate limiting timestamp tracking
from pathlib import Path

import fitz  # PyMuPDF
import google.api_core.exceptions
import google.generativeai as genai
from dotenv import load_dotenv
from natsort import natsorted  # Import natsorted
from rich.align import Align  # For centering

# Rich imports for console formatting
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule  # For separators
from rich.text import Text

# Load environment variables from .env file, if it exists
load_dotenv()

# --- Constants ---

NOTES_DIR_PATH_STR = os.environ.get("NOTES_DIR_PATH")
NOTES_DIR = Path(NOTES_DIR_PATH_STR) if NOTES_DIR_PATH_STR else None
SYS_PROMPT_FILE = Path("sysprompt.txt")
BACKGROUND_FILE = Path("background.txt")  # File to be ignored by git
HISTORY_DIR = Path("history")  # Directory to store chat logs, to be ignored by git
GITIGNORE_FILE = Path(".gitignore")
API_KEY = os.environ.get("GEMINI_API_KEY")

# Warning: 'gemini-2.5-pro-exp-03-25' might be experimental. Use 'gemini-pro' or 'gemini-1.5-pro-latest' if issues arise.
MODEL_NAME = "gemini-2.5-pro-exp-03-25"

# Rate Limiting: Maximum number of requests allowed per minute
REQUESTS_PER_MINUTE = 2  # Adjusted from user's code

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

# --- Initialize Rich Console ---
console = Console()

# --- Function Definitions ---


def EnsureGitignore():
    """
    Checks if .gitignore exists and ensures specific files/directories are ignored.

    Creates .gitignore if it doesn't exist. Appends necessary ignore rules
    for sensitive files like background info and chat history. Uses Rich output.
    """

    ignorePatterns = {
        BACKGROUND_FILE.name,  # e.g., "background.txt"
        f"{HISTORY_DIR.name}/",  # e.g., "history/" - trailing slash for directory
        ".env",  # Good practice to ignore environment file
    }
    patternsToAdd = set()
    needsUpdate = False

    #
    console.print(
        f"Checking [cyan]{GITIGNORE_FILE}[/] for necessary ignore patterns..."
    )

    #
    try:
        #
        if GITIGNORE_FILE.is_file():
            #
            # Read existing patterns
            with open(GITIGNORE_FILE, "r", encoding="utf-8") as f:
                #
                existingPatterns = {
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                }

            #
            # Determine which patterns are missing
            patternsToAdd = ignorePatterns - existingPatterns

            #
            if patternsToAdd:
                #
                needsUpdate = True
                console.print(
                    f"  [yellow]Will add missing patterns:[/yellow] {', '.join(patternsToAdd)}"
                )
            #
            else:
                #
                console.print(
                    f"  [green]Already contains the necessary patterns.[/green]"
                )

        #
        else:
            #
            # File doesn't exist, need to create it and add all patterns
            needsUpdate = True
            patternsToAdd = ignorePatterns
            console.print(
                f"  [yellow]{GITIGNORE_FILE} not found. Will create it.[/yellow]"
            )

        #
        # If updates are needed, append to or create the file
        if needsUpdate:
            #
            with open(GITIGNORE_FILE, "a", encoding="utf-8") as f:
                #
                # Add a header if the file is newly created or being added to significantly
                if (
                    not GITIGNORE_FILE.exists() or patternsToAdd == ignorePatterns
                ):  # Crude check if file was just created
                    f.write("\n# Ignore patterns added by AI Therapist script\n")
                elif (
                    patternsToAdd
                ):  # Only add header if adding something new to existing file
                    f.write("\n# Added by AI Therapist script\n")

                #
                for pattern in sorted(list(patternsToAdd)):  # Sort for consistent order
                    #
                    f.write(f"{pattern}\n")
            #
            console.print(f"  [green]{GITIGNORE_FILE} updated successfully.[/green]")

    except IOError as e:
        #
        console.print(
            f"[bold red]Error accessing or modifying {GITIGNORE_FILE}:[/] {e}"
        )
        console.print("  Skipping .gitignore check/update.")
    #
    except Exception as e:
        #
        console.print(
            f"[bold red]An unexpected error occurred during .gitignore check:[/] {e}"
        )
        console.print("  Skipping .gitignore check/update.")

    #
    # Add a rule after the check is complete
    console.print(Rule("Gitignore Check Complete", style="dim blue"))


def LoadTextFile(filePath: Path, fileDescription: str) -> str | None:
    """
    Loads text content from a specified file. Uses Rich for output.

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

    #
    console.print(f"Loading {fileDescription}: [cyan]{filePath}[/]", style="dim")

    #
    try:
        #
        fileContent = filePath.read_text(encoding="utf-8")
        #
        # console.print(f"  [green]Successfully loaded.[/]") # Keep output less verbose

        #
        return fileContent

    except FileNotFoundError:
        #
        console.print(f"  [bold red]Error:[/] File not found.")

        #
        return None

    except IOError as e:
        #
        console.print(f"  [bold red]Error reading file:[/] {e}")

        #
        return None


def LoadTherapyNotes(directory: Path | None) -> str:
    """
    Loads and extracts text from all PDF files in the specified directory.
    Uses Rich Progress bar for visual feedback.

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
        console.print(
            "[yellow]Info:[/] NOTES_DIR_PATH env var not set. No therapy notes loaded."
        )

        #
        return ""

    #
    if not directory.is_dir():
        #
        console.print(
            f"[bold red]Error:[/] Notes directory not found/invalid: [cyan]{directory}[/]. No notes loaded."
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
        console.print(f"[yellow]Info:[/] No PDF files found in [cyan]{directory}[/].")

        #
        return ""

    #
    processedCount = 0
    errorsEncountered = 0

    #
    # Use Rich Progress for PDF processing visualization
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),  # Set bar width to expand
        TaskProgressColumn(),  # Percentage for the current task
        TextColumn("[cyan]({task.completed}/{task.total})[/]"),
        console=console,
    ) as progress:
        #
        pdfTask = progress.add_task(
            f"Processing {len(pdfFiles)} PDFs in [cyan]{directory.name}[/]",
            total=len(pdfFiles),
        )

        #
        for pdfPath in pdfFiles:
            #
            progress.update(pdfTask, description=f"Processing [cyan]{pdfPath.name}[/]")

            #
            try:
                #
                if not pdfPath.is_file():
                    # console.print(f"  [yellow]Warning:[/] Skipping [cyan]{pdfPath.name}[/], not a valid file.") # Too verbose with progress bar
                    progress.console.print(
                        f"[dim]  Skipping non-file: {pdfPath.name}[/dim]"
                    )
                    continue

                #
                doc = fitz.open(pdfPath)
                pdfText = "".join(
                    page.get_text("text")
                    for page in doc.pages()
                    if page.get_text("text")
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
                    # console.print(f"  [green]Successfully extracted text from {pdfPath.name}.[/]") # Too verbose
                    processedCount += 1
                #
                else:
                    #
                    # console.print(f"  [yellow]Warning:[/] No text extracted from [cyan]{pdfPath.name}[/].") # Too verbose
                    progress.console.print(
                        f"[yellow]  No text in: {pdfPath.name}[/yellow]"
                    )

            except fitz.errors.FileDataError:
                #
                progress.console.print(
                    f"[red]  Error (corrupt/pwd?): {pdfPath.name}[/red]"
                )
                errorsEncountered += 1
            #
            except Exception as e:
                #
                progress.console.print(
                    f"[red]  Error processing {pdfPath.name}: {e}[/red]"
                )
                errorsEncountered += 1
            #
            finally:
                #
                progress.update(
                    pdfTask, advance=1
                )  # Advance progress bar regardless of success/failure

    #
    # Print summary after progress bar finishes
    if not notesTexts:
        #
        console.print(
            "[bold yellow]Warning:[/] Could not extract text from any valid PDF files."
        )

        #
        return ""

    #
    summary_color = "green" if errorsEncountered == 0 else "yellow"
    console.print(
        f"[{summary_color}]Successfully combined therapy notes from {processedCount} PDF(s).[/]"
        f"{f' ([red]{errorsEncountered} errors[/])' if errorsEncountered > 0 else ''}"
    )

    #
    # Combine all extracted notes into a single string
    combinedNotes = "\n\n".join(notesTexts)

    #
    return combinedNotes


def main():
    """
    Main function to run the AI Therapist application with multi-turn chat and logging.
    """

    console.print(Rule("[bold cyan]AI Therapist Initializing[/]", style="cyan"))

    # --- Ensure Gitignore Rules ---
    EnsureGitignore()  # Check/update .gitignore before proceeding

    # --- Validate Environment Variables & Setup History Dir ---

    #
    if not API_KEY:
        #
        console.print(
            Panel(
                "[bold red]Fatal Error: GEMINI_API_KEY environment variable not set.[/]\nPlease set the environment variable (or use a .env file) and restart.",
                title="Error",
                border_style="red",
            )
        )

        #
        return

    #
    if not NOTES_DIR_PATH_STR:  # Check if the path string was retrieved
        #
        console.print(
            "[yellow]Warning:[/] 'NOTES_DIR_PATH' environment variable not set or invalid."
        )
        # NOTES_DIR will be None, LoadTherapyNotes handles this.

    #
    # Create history directory if it doesn't exist
    logFile = None  # Initialize logFile to None
    logFilename = None  # Initialize logFilename to None
    #
    try:
        #
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    #
    except OSError as e:
        #
        console.print(
            f"[bold red]Error creating history directory {HISTORY_DIR}:[/] {e}"
        )
        # Decide if this is fatal or just proceed without logging
        console.print("[yellow]Proceeding without chat logging.[/]")
        # logFile remains None
    #
    else:
        # --- Setup Logging ---
        # Generate a unique filename for this session's log
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logFilename = HISTORY_DIR / f"chat_log_{timestamp}.txt"
        console.print(f"Logging chat session to: [cyan]{logFilename}[/]")
        # Open the log file in append mode, using 'utf-8' encoding
        try:
            #
            logFile = open(logFilename, "a", encoding="utf-8")
        #
        except IOError as e:
            #
            console.print(f"[bold red]Error opening log file {logFilename}:[/] {e}")
            logFile = None  # Ensure logFile is None if opening fails

    console.print(Rule("Loading Context", style="dim"))

    # --- Load Static Context ---

    #
    baseSystemPrompt = LoadTextFile(SYS_PROMPT_FILE, "system prompt")

    #
    if baseSystemPrompt is None:
        #
        console.print(
            Panel(
                "[bold red]Fatal Error: Could not load system prompt. Exiting.[/]",
                title="Error",
                border_style="red",
            )
        )
        if logFile:
            logFile.close()  # Close log file if open

        #
        return

    #
    userBackground = LoadTextFile(BACKGROUND_FILE, "user background")

    #
    if userBackground is None:
        #
        console.print(
            "[yellow]Warning:[/] Could not load user background file. Using fallback."
        )
        userBackground = "No user background information was loaded."

    #
    therapyNotesText = LoadTherapyNotes(NOTES_DIR)  # Handles None NOTES_DIR

    # --- Construct the SINGLE System Instruction ---
    # Combine all static context into one block for the model's system instruction

    #
    systemInstructionParts = [baseSystemPrompt]

    #
    systemInstructionParts.append(
        "\n\n--- USER BACKGROUND INFORMATION ---\n"
        "Use this information to personalize responses and understand context.\n\n"
        f"{userBackground}\n\n"
        "--- END USER BACKGROUND INFORMATION ---"
    )

    #
    if therapyNotesText:
        #
        systemInstructionParts.append(
            "\n\n--- BEGIN THERAPY NOTES HISTORY ---\n"
            "Use these notes (prioritizing over background if conflicting) for context.\n\n"
            f"{therapyNotesText}\n\n"
            "--- END THERAPY NOTES HISTORY ---"
        )
    #
    else:
        #
        systemInstructionParts.append(
            "\n\n--- THERAPY NOTES HISTORY ---\n"
            "No therapy notes loaded. Base responses on background and conversation."
            "\n--- END THERAPY NOTES HISTORY ---"
        )

    #
    # Append Ethical Guidelines Reminder
    systemInstructionParts.append(
        "\n\n--- IMPORTANT REMINDERS ---\n"
        "1. Role: AI assistant, NOT licensed therapist. No diagnoses/treatment.\n"
        "2. Ethics: Adhere to initial system prompt guidelines.\n"
        "3. Crisis: If user mentions self-harm/suicide/abuse, respond empathetically, "
        "urge professional help, provide crisis resources (e.g., 988 US). Do not manage crisis.\n"
        "4. Context: Refer to User Background/Therapy Notes.\n"
        "--- END IMPORTANT REMINDERS ---"
    )

    #
    combinedSystemInstruction = "".join(systemInstructionParts)

    # --- Log Initial Context (Optional) ---
    if logFile:
        #
        logFile.write(f"--- Session Start: {timestamp} ---\n")
        logFile.write(f"Model: {MODEL_NAME}\n")
        logFile.write(f"Rate Limit: {REQUESTS_PER_MINUTE}/min\n")
        logFile.write("--- System Instruction Sent to Model ---\n")
        logFile.write(combinedSystemInstruction)
        logFile.write("\n--- End System Instruction ---\n\n")
        logFile.flush()  # Ensure initial info is written

    console.print(Rule("Initializing AI Model", style="dim"))

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
            system_instruction=combinedSystemInstruction,
        )

        #
        # Start a chat session (history is managed internally)
        chat = model.start_chat(history=[])  # Start with empty user/model turn history

        #
        # Display centered "Ready" panel
        readyPanel = Panel(
            f"[bold green]AI Therapist Ready[/]\n"
            f"Model: [cyan]{MODEL_NAME}[/]\n"
            f"Rate Limit: {REQUESTS_PER_MINUTE} requests per minute.\n\n"
            "Type '[bold]quit[/]' or '[bold]exit[/]' to end the session.\n"
            "[dim]Disclaimer: AI for support, not professional therapy/medical advice.[/]",
            title="Session Started",
            border_style="green",
            expand=False,
        )
        console.print(Align.center(readyPanel))

    except Exception as e:
        #
        errorMsgContent = f"Fatal Error during AI Initialization: {e}\nCheck API key, model name, and network connection."
        errorMsg = Panel(
            f"[bold red]{errorMsgContent}[/]",
            title="Initialization Error",
            border_style="red",
        )
        console.print(errorMsg)
        if logFile:
            #
            logFile.write(
                errorMsgContent + "\n--- SESSION ENDED DUE TO INIT ERROR ---\n"
            )
            logFile.close()

        #
        return

    # --- Conversation Loop ---

    # Initialize deque to store timestamps of messages sent
    messageTimestamps = deque()

    #
    try:  # Main loop try block
        #
        while True:
            #
            # Get user input
            try:
                # Use console.input for rich prompt
                userInput = console.input(Text("\nYou: ", style="bold blue")).strip()

                #
                if not userInput:  # Handle empty input
                    continue

            except EOFError:
                #
                console.print(
                    "\n[yellow]AI Therapist: Input stream closed. Ending session.[/]"
                )
                if logFile:
                    logFile.write("\n--- Input stream closed by user ---\n")

                #
                break
            #
            except KeyboardInterrupt:  # Catch Ctrl+C during input
                #
                console.print(
                    "\n\n[yellow]AI Therapist: Session interrupted by user. Goodbye.[/]"
                )
                if logFile:
                    logFile.write(
                        "\n--- Session interrupted by user (KeyboardInterrupt during input) ---\n"
                    )

                #
                break  # Exit the loop cleanly

            #
            # Check for exit command
            if userInput.lower() in ["quit", "exit"]:
                #
                console.print(
                    "\n[yellow]AI Therapist: Ending session as requested. Take care.[/]"
                )
                if logFile:
                    logFile.write(
                        "\n--- Session ended by user command ('quit'/'exit') ---\n"
                    )

                #
                break

            # --- Log User Input ---
            if logFile:
                #
                logFile.write(
                    f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] You: {userInput}\n"
                )
                logFile.flush()

            # --- Rate Limiting Check ---

            #
            currentTime = time.time()

            # Remove timestamps older than 60 seconds from the left
            while messageTimestamps and currentTime - messageTimestamps[0] > 60:
                #
                messageTimestamps.popleft()

            #
            # Check if the number of recent messages meets or exceeds the limit
            if len(messageTimestamps) >= REQUESTS_PER_MINUTE:
                #
                timeSinceOldest = currentTime - messageTimestamps[0]
                # Calculate time needed to wait until the oldest message is > 60s old
                waitTime = 60 - timeSinceOldest + 0.1  # Add small buffer
                waitMsg = (
                    f"Rate limit reached ({REQUESTS_PER_MINUTE}/min). Please wait..."
                )

                #
                console.print(
                    Panel(
                        f"[bold yellow]{waitMsg}[/]",
                        border_style="yellow",
                        expand=False,
                    )
                )
                if logFile:
                    logFile.write(
                        f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] System: {waitMsg.strip()}\n"
                    )

                #
                # Display rich progress bar timer
                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None),  # Use full width
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console,
                    transient=True,  # Clear the progress bar when done
                ) as progress:
                    #
                    waitTask = progress.add_task(
                        "[yellow]Time until next message:[/]", total=waitTime
                    )
                    while not progress.finished:
                        #
                        progress.update(waitTask, advance=0.1)
                        time.sleep(0.1)

                #
                if logFile:
                    logFile.write(f"  (Rate limit wait finished)\n")

                #
                # Update current time after waiting
                currentTime = time.time()
                # Re-clean timestamps after waiting (optional but good practice)
                while messageTimestamps and currentTime - messageTimestamps[0] > 60:
                    #
                    messageTimestamps.popleft()

            # --- Send Message and Get Response ---

            #
            console.print(Text("\nAI Therapist: (Thinking...)", style="italic dim"))
            aiResponseText = None  # Initialize in case of error before assignment

            #
            try:
                #
                # Record the timestamp *before* sending the message
                messageTimestamps.append(time.time())

                #
                # Send message - chat history is automatically updated by the object
                response = chat.send_message(userInput)

                # --- Process Response ---
                # Check for blocked content or other issues

                #
                if not response.parts:
                    #
                    blockReason = "Unknown"  # Renamed variable

                    #
                    if (
                        hasattr(response, "prompt_feedback")
                        and response.prompt_feedback.block_reason
                    ):
                        #
                        blockReason = (
                            response.prompt_feedback.block_reason.name
                        )  # Use .name for enum
                    #
                    elif (
                        hasattr(response, "candidates")
                        and response.candidates
                        and response.candidates[0].finish_reason.name != "STOP"
                    ):
                        #
                        blockReason = f"Finish Reason: {response.candidates[0].finish_reason.name}"

                    #
                    warningMsgContent = (
                        f"AI response empty/blocked. Reason: {blockReason}"
                    )
                    warningMsg = Panel(
                        f"[bold yellow]Warning:[/] {warningMsgContent}",
                        title="Warning",
                        border_style="yellow",
                    )
                    console.print(warningMsg)
                    aiResponseText = (
                        "I apologize, but I encountered an issue or the content was blocked. "
                        "Could you rephrase or try a different topic?"
                    )
                    if logFile:
                        logFile.write(
                            f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] System: {warningMsgContent}\n"
                        )

                #
                else:
                    #
                    aiResponseText = response.text

                #
                # Print the AI's response using Markdown rendering within a Panel
                console.print(
                    Panel(
                        Markdown(aiResponseText),
                        title="AI Therapist",
                        border_style="cyan",
                        expand=False,  # Don't force panel to full width
                    )
                )

                # --- Log AI Response ---
                if logFile:
                    #
                    logFile.write(
                        f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] AI Therapist: {aiResponseText}\n"
                    )
                    logFile.flush()

            # --- API Error Handling within the loop ---
            except google.api_core.exceptions.PermissionDenied as e:
                #
                errorMsgContent = f"API Error: Permission Denied. Check API key/model access. Details: {e}"
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]",
                    title="API Error",
                    border_style="red",
                )
                console.print(errorMsg)
                if logFile:
                    logFile.write(
                        f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] System Error: {errorMsgContent}\n"
                    )
                # Decide whether to break or allow retry
                # break
            #
            except google.api_core.exceptions.NotFound as e:
                #
                errorMsgContent = f"API Error: Model '{MODEL_NAME}' Not Found or unavailable. Details: {e}"
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]",
                    title="API Error",
                    border_style="red",
                )
                console.print(errorMsg)
                if logFile:
                    logFile.write(
                        f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] System Error: {errorMsgContent}\n--- SESSION ENDED DUE TO API ERROR ---\n"
                    )

                #
                break  # Likely fatal for this session
            #
            except google.api_core.exceptions.ResourceExhausted as e:
                #
                errorMsgContent = f"API Error: Quota Exceeded. Details: {e}"
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]",
                    title="API Error",
                    border_style="red",
                )
                console.print(errorMsg)
                # Remove the timestamp for the failed request
                if messageTimestamps:
                    messageTimestamps.pop()
                if logFile:
                    logFile.write(
                        f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] System Error: {errorMsgContent}\n--- SESSION ENDED DUE TO API ERROR ---\n"
                    )

                #
                break
            #
            except google.api_core.exceptions.InvalidArgument as e:
                #
                errorMsgContent = (
                    f"API Error: Invalid Argument (check prompt/content). Details: {e}"
                )
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]",
                    title="API Error",
                    border_style="red",
                )
                console.print(errorMsg)
                # Remove the timestamp for the failed request
                if messageTimestamps:
                    messageTimestamps.pop()
                if logFile:
                    logFile.write(
                        f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] System Error: {errorMsgContent}\n"
                    )
                # Log problematic input if needed (beware sensitive data)
                # if logFile: logFile.write(f"Input causing error: {userInput}\n")
            #
            except Exception as e:
                #
                errorMsgContent = f"Error during AI response generation: {e}"
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]", title="Error", border_style="red"
                )
                console.print(errorMsg)
                # Remove the timestamp for the failed request
                if messageTimestamps:
                    messageTimestamps.pop()
                if logFile:
                    logFile.write(
                        f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] System Error: {errorMsgContent}\n"
                    )
                # General error, maybe allow user to try again

    # --- General Error Handling & Cleanup ---
    # KeyboardInterrupt is now caught during input()
    except Exception as e:  # Catch other unexpected errors in the loop
        #
        errorMsgContent = f"An unexpected error occurred in the main loop: {e}"
        errorMsg = Panel(
            f"[bold red]{errorMsgContent}[/]",
            title="Unexpected Error",
            border_style="red",
        )
        console.print(errorMsg)
        if logFile:
            logFile.write(
                f"\n{errorMsgContent}\n--- SESSION ENDED DUE TO UNEXPECTED ERROR ---\n"
            )
    #
    finally:
        #
        # Ensure the log file is closed properly
        if logFile:
            #
            logFile.write(
                f"\n--- Session End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
            )
            logFile.close()
            if logFilename:  # Check if logFilename was set
                console.print(f"Chat log saved to: [cyan]{logFilename}[/]")
        #
        console.print(Rule("[bold cyan]AI Therapist Session Ended[/]", style="cyan"))


# --- Main Execution ---

#
if __name__ == "__main__":
    #
    main()
