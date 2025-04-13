"""
AI Therapist Application using Google Generative AI (Multi-Turn Chat).

This script implements a conversational AI therapist that uses Google's
Generative AI API (e.g., Gemini) and maintains conversation history within
a single session. It loads a system prompt, user background, therapy notes,
past conversation logs, implements rate limiting, logs chat sessions,
ensures .gitignore rules, and engages in a therapeutic conversation
with enhanced console output using Rich.

Prerequisites:
    - Python 3.10+
    - Google Generative AI API Key set as an environment variable: GEMINI_API_KEY
    - Path to therapy notes directory set as an environment variable: NOTES_DIR_PATH
    - Required libraries installed:
        pip install google-generativeai PyMuPDF python-dotenv pathlib natsort rich pytz

Setup:
    1. Create `sysprompt.txt` (core AI instructions).
    2. Create `background.txt` (user background info).
    3. Create `.env` file (optional) or set environment variables:
       GEMINI_API_KEY='YOUR_API_KEY'
       NOTES_DIR_PATH='/full/path/to/your/notes'
    4. Ensure the notes directory exists.
    5. A 'history' directory containing 'pre_chat_logs' and 'chat_logs' subdirectories
       will be created automatically for chat logs.
    6. A '.gitignore' file will be checked/created to ignore sensitive files.

Execution:
    python therapist.py
"""

import datetime  # For logging filenames
import os
import time  # For rate limiting and logging
from collections import deque  # For rate limiting timestamp tracking
from pathlib import Path

import fitz  # PyMuPDF
import google.api_core.exceptions
import google.generativeai as genai
import pytz
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
HISTORY_DIR = Path("history")  # Base directory for logs
PRE_LOG_SUBDIR = HISTORY_DIR / "pre_chat_logs"  # Subdirectory for pre-chat logs
CHAT_LOG_SUBDIR = HISTORY_DIR / "chat_logs"  # Subdirectory for chat logs
GITIGNORE_FILE = Path(".gitignore")
API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL_NAME = "gemini-2.5-pro-exp-03-25"

# Rate Limiting: Maximum number of requests allowed per minute
REQUESTS_PER_MINUTE = 5

# TODO: 2.5 Pro daily limit is 25. Keep track?

# Timestamp format for detailed log entries (user/AI messages)
LOG_DETAIL_TIMESTAMP_FORMAT = "%Y-%m-%d %A %H:%M:%S"
# Timestamp format for log file start/end markers
LOG_FILE_TIMESTAMP_FORMAT = "%Y-%m-%d %A %H:%M:%S"

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

    console.print(
        f"Checking [cyan]{GITIGNORE_FILE}[/] for necessary ignore patterns..."
    )

    try:

        if GITIGNORE_FILE.is_file():

            # Read existing patterns
            with open(GITIGNORE_FILE, "r", encoding="utf-8") as f:

                existingPatterns = {
                    line.strip()
                    for line in f
                    if line.strip() and not line.strip().startswith("#")
                }

            # Determine which patterns are missing
            patternsToAdd = ignorePatterns - existingPatterns

            if patternsToAdd:

                needsUpdate = True
                console.print(
                    f"  [yellow]Will add missing patterns:[/yellow] {', '.join(patternsToAdd)}"
                )

            else:

                console.print(
                    f"  [green]Already contains the necessary patterns.[/green]"
                )

        else:

            # File doesn't exist, need to create it and add all patterns
            needsUpdate = True
            patternsToAdd = ignorePatterns
            console.print(
                f"  [yellow]{GITIGNORE_FILE} not found. Will create it.[/yellow]"
            )

        # If updates are needed, append to or create the file
        if needsUpdate:

            with open(GITIGNORE_FILE, "a", encoding="utf-8") as f:

                # Add a header if the file is newly created or being added to significantly
                if (
                    not GITIGNORE_FILE.exists() or patternsToAdd == ignorePatterns
                ):  # Crude check if file was just created
                    f.write("\n# Ignore patterns added by AI Therapist script\n")
                elif (
                    patternsToAdd
                ):  # Only add header if adding something new to existing file
                    f.write("\n# Added by AI Therapist script\n")

                for pattern in sorted(list(patternsToAdd)):  # Sort for consistent order

                    f.write(f"{pattern}\n")

            console.print(f"  [green]{GITIGNORE_FILE} updated successfully.[/green]")

    except IOError as e:

        console.print(
            f"[bold red]Error accessing or modifying {GITIGNORE_FILE}:[/] {e}"
        )
        console.print("  Skipping .gitignore check/update.")

    except Exception as e:

        console.print(
            f"[bold red]An unexpected error occurred during .gitignore check:[/] {e}"
        )
        console.print("  Skipping .gitignore check/update.")

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

    console.print(f"Loading {fileDescription}: [cyan]{filePath}[/]", style="dim")

    try:

        fileContent = filePath.read_text(encoding="utf-8")

        # console.print(f"  [green]Successfully loaded.[/]") # Keep output less verbose

        return fileContent

    except FileNotFoundError:

        console.print(f"  [bold red]Error:[/] File not found.")

        return None

    except IOError as e:

        console.print(f"  [bold red]Error reading file:[/] {e}")

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

    if directory is None:

        console.print(
            "[yellow]Info:[/] NOTES_DIR_PATH env var not set. No therapy notes loaded."
        )

        return ""

    if not directory.is_dir():

        console.print(
            f"[bold red]Error:[/] Notes directory not found/invalid: [cyan]{directory}[/]. No notes loaded."
        )

        return ""

    notesTexts = []
    # Use natsorted for natural sorting (e.g., file2.pdf before file10.pdf)
    pdfFiles = natsorted(list(directory.glob("*.pdf")))

    if not pdfFiles:

        console.print(f"[yellow]Info:[/] No PDF files found in [cyan]{directory}[/].")

        return ""

    processedCount = 0
    errorsEncountered = 0

    # Use Rich Progress for PDF processing visualization
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),  # Set bar width to expand
        TaskProgressColumn(),  # Percentage for the current task
        TextColumn("[cyan]({task.completed}/{task.total})[/]"),
        console=console,
    ) as progress:

        pdfTask = progress.add_task(
            f"Processing {len(pdfFiles)} PDFs in [cyan]{directory.name}[/]",
            total=len(pdfFiles),
        )

        for pdfPath in pdfFiles:

            progress.update(pdfTask, description=f"Processing [cyan]{pdfPath.name}[/]")

            try:

                if not pdfPath.is_file():
                    # console.print(f"  [yellow]Warning:[/] Skipping [cyan]{pdfPath.name}[/], not a valid file.") # Too verbose with progress bar
                    progress.console.print(
                        f"[dim]  Skipping non-file: {pdfPath.name}[/dim]"
                    )
                    continue

                doc = fitz.open(pdfPath)
                pdfText = "".join(
                    page.get_text("text")
                    for page in doc.pages()
                    if page.get_text("text")
                )
                doc.close()

                pdfTextStripped = pdfText.strip()

                if pdfTextStripped:

                    notesTexts.append(
                        f"--- Start Notes: {pdfPath.name} ---\n{pdfTextStripped}\n--- End Notes: {pdfPath.name} ---"
                    )
                    # console.print(f"  [green]Successfully extracted text from {pdfPath.name}.[/]") # Too verbose
                    processedCount += 1

                else:

                    # console.print(f"  [yellow]Warning:[/] No text extracted from [cyan]{pdfPath.name}[/].") # Too verbose
                    progress.console.print(
                        f"[yellow]  No text in: {pdfPath.name}[/yellow]"
                    )

            except fitz.errors.FileDataError:

                progress.console.print(
                    f"[red]  Error (corrupt/pwd?): {pdfPath.name}[/red]"
                )
                errorsEncountered += 1

            except Exception as e:

                progress.console.print(
                    f"[red]  Error processing {pdfPath.name}: {e}[/red]"
                )
                errorsEncountered += 1

            finally:

                progress.update(
                    pdfTask, advance=1
                )  # Advance progress bar regardless of success/failure

    # Print summary after progress bar finishes
    if not notesTexts:

        console.print(
            "[bold yellow]Warning:[/] Could not extract text from any valid PDF files."
        )

        return ""

    summary_color = "green" if errorsEncountered == 0 else "yellow"
    console.print(
        f"[{summary_color}]Successfully combined therapy notes from {processedCount} PDF(s).[/]"
        f"{f' ([red]{errorsEncountered} errors[/])' if errorsEncountered > 0 else ''}"
    )

    # Combine all extracted notes into a single string
    combinedNotes = "\n\n".join(notesTexts)

    return combinedNotes


def LoadPastConversations(chat_log_dir: Path) -> str:
    """
    Loads and combines content from previous chat log files.

    Parameters
    ----------
    chat_log_dir : Path
        The path object pointing to the directory containing chat log files.

    Returns
    -------
    str
        A single string containing the combined text from all chat logs,
        separated by dividers. Returns an empty string if the directory
        doesn't exist, is empty, or an error occurs.
    """
    console.print(
        f"Loading past conversations from: [cyan]{chat_log_dir}[/]", style="dim"
    )
    if not chat_log_dir.is_dir():
        console.print(
            f"  [yellow]Info:[/] Chat log directory not found. No past conversations loaded."
        )
        return ""

    # Find chat log files, sort them naturally by filename (which includes timestamp)
    log_files = natsorted(
        [f for f in chat_log_dir.glob("chat_log_*.txt") if f.is_file()]
    )

    if not log_files:
        console.print(
            f"  [yellow]Info:[/] No past chat logs found in [cyan]{chat_log_dir}[/]."
        )
        return ""

    combined_history = []
    errors_encountered = 0
    logs_loaded = 0

    console.print(f"  Found {len(log_files)} past chat log(s). Combining...")

    for log_path in log_files:
        try:
            log_content = log_path.read_text(encoding="utf-8").strip()
            if log_content:
                # Add separators for clarity in the combined context
                combined_history.append(
                    f"--- Start Past Conversation: {log_path.name} ---\n"
                    f"{log_content}\n"
                    f"--- End Past Conversation: {log_path.name} ---"
                )
                logs_loaded += 1
            else:
                console.print(f"  [dim]Skipping empty log file: {log_path.name}[/dim]")
        except IOError as e:
            console.print(
                f"  [bold red]Error reading past log file {log_path.name}:[/] {e}"
            )
            errors_encountered += 1
        except Exception as e:
            console.print(
                f"  [bold red]Unexpected error processing log file {log_path.name}:[/] {e}"
            )
            errors_encountered += 1

    if not combined_history:
        console.print(
            "  [yellow]Warning:[/] Could not load content from any past chat logs."
        )
        return ""

    summary_color = "green" if errors_encountered == 0 else "yellow"
    console.print(
        f"  [{summary_color}]Successfully loaded content from {logs_loaded} past conversation(s).[/]"
        f"{f' ([red]{errors_encountered} errors[/])' if errors_encountered > 0 else ''}"
    )

    # Join all loaded histories with double newlines
    return "\n\n".join(combined_history)


def main():
    """
    Main function to run the AI Therapist application with multi-turn chat and logging.
    """
    start_time = datetime.datetime.now()  # Get start time for log headers
    formatted_start_time = start_time.strftime(LOG_FILE_TIMESTAMP_FORMAT)

    console.print(Rule("[bold cyan]AI Therapist Initializing[/]", style="cyan"))

    # --- Ensure Gitignore Rules ---
    EnsureGitignore()  # Check/update .gitignore before proceeding

    # --- Validate Environment Variables & Setup History Dir ---

    if not API_KEY:

        console.print(
            Panel(
                "[bold red]Fatal Error: GEMINI_API_KEY environment variable not set.[/]\nPlease set the environment variable (or use a .env file) and restart.",
                title="Error",
                border_style="red",
            )
        )

        return

    if not NOTES_DIR_PATH_STR:  # Check if the path string was retrieved

        console.print(
            "[yellow]Warning:[/] 'NOTES_DIR_PATH' environment variable not set or invalid."
        )
        # NOTES_DIR will be None, LoadTherapyNotes handles this.

    # --- Setup Logging ---
    preLogFile = None  # Initialize pre-log file handle
    chatLogFile = None  # Initialize chat log file handle
    preLogFilename = None  # Initialize pre-log filename
    chatLogFilename = None  # Initialize chat log filename
    logDirsCreated = False

    try:
        # Create base history directory and subdirectories
        PRE_LOG_SUBDIR.mkdir(parents=True, exist_ok=True)
        CHAT_LOG_SUBDIR.mkdir(parents=True, exist_ok=True)
        logDirsCreated = True

    except OSError as e:

        console.print(
            f"[bold red]Error creating history directories under {HISTORY_DIR}:[/] {e}"
        )
        console.print("[yellow]Proceeding without any file logging.[/]")

    if logDirsCreated:
        # Generate unique filenames using the start time's timestamp part
        timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
        preLogFilename = PRE_LOG_SUBDIR / f"pre_chat_log_{timestamp_str}.txt"
        chatLogFilename = CHAT_LOG_SUBDIR / f"chat_log_{timestamp_str}.txt"

        console.print(f"Attempting to log pre-chat info to: [cyan]{preLogFilename}[/]")
        console.print(
            f"Attempting to log chat conversation to: [cyan]{chatLogFilename}[/]"
        )

        # Open the pre-log file
        try:
            preLogFile = open(preLogFilename, "w", encoding="utf-8")
            # Use formatted start time for the header
            preLogFile.write(f"--- Pre-Chat Log Start: {formatted_start_time} ---\n")
            preLogFile.write(f"Model: {MODEL_NAME}\n")
            preLogFile.write(f"Rate Limit: {REQUESTS_PER_MINUTE}/min\n")
            preLogFile.flush()
        except IOError as e:
            console.print(
                f"[bold red]Error opening pre-log file {preLogFilename}:[/] {e}"
            )
            preLogFile = None
            preLogFilename = None

    console.print(Rule("Loading Context", style="dim"))

    # --- Load Static Context ---

    baseSystemPrompt = LoadTextFile(SYS_PROMPT_FILE, "system prompt")

    if baseSystemPrompt is None:

        console.print(
            Panel(
                "[bold red]Fatal Error: Could not load system prompt. Exiting.[/]",
                title="Error",
                border_style="red",
            )
        )
        if preLogFile:
            preLogFile.write("\n--- FATAL ERROR: Could not load system prompt. ---\n")
            preLogFile.close()

        return

    def GetFormattedDateTime() -> str:
        """
        Gets the current date and time formatted with time zone, non-24-hour clock, and day of the week.

        Returns
        -------
        str
            The formatted date and time string.
        """

        # TODO: Replace with desired timezone
        localTimezone = pytz.timezone("America/New_York")
        now = datetime.datetime.now(localTimezone)
        return now.strftime("%A, %I:%M %p %Z on %B %d, %Y")

    formattedDateTime = GetFormattedDateTime()

    # Add formatted date and time to the base system prompt
    baseSystemPrompt = (
        f"Current Date and Time: {formattedDateTime}\n\n{baseSystemPrompt}"
    )

    userBackground = LoadTextFile(BACKGROUND_FILE, "user background")

    if userBackground is None:

        console.print(
            "[yellow]Warning:[/] Could not load user background file. Using fallback."
        )
        userBackground = "No user background information was loaded."

    therapyNotesText = LoadTherapyNotes(NOTES_DIR)

    # --- Load Past Conversation History ---
    pastConversationsText = LoadPastConversations(CHAT_LOG_SUBDIR)

    # --- Construct the SINGLE System Instruction ---
    systemInstructionParts = [baseSystemPrompt]

    systemInstructionParts.append(
        "\n\n--- USER BACKGROUND INFORMATION ---\n"
        "Use this information to personalize responses and understand context.\n\n"
        f"{userBackground}\n\n"
        "--- END USER BACKGROUND INFORMATION ---"
    )

    if therapyNotesText:
        systemInstructionParts.append(
            "\n\n--- BEGIN THERAPY NOTES HISTORY ---\n"
            "Use these notes (prioritizing over background if conflicting) for context.\n\n"
            f"{therapyNotesText}\n\n"
            "--- END THERAPY NOTES HISTORY ---"
        )
    else:
        systemInstructionParts.append(
            "\n\n--- THERAPY NOTES HISTORY ---\n"
            "No therapy notes loaded."
            "\n--- END THERAPY NOTES HISTORY ---"
        )

    # Add Past Conversations if loaded
    if pastConversationsText:
        systemInstructionParts.append(
            "\n\n--- BEGIN PAST CONVERSATION HISTORY ---\n"
            "This section contains logs from previous chat sessions. Use this for long-term context.\n\n"
            f"{pastConversationsText}\n\n"
            "--- END PAST CONVERSATION HISTORY ---"
        )
    else:
        systemInstructionParts.append(
            "\n\n--- PAST CONVERSATION HISTORY ---\n"
            "No past conversation history loaded."
            "\n--- END PAST CONVERSATION HISTORY ---"
        )

    # Append Ethical Guidelines Reminder (Keep this towards the end)
    systemInstructionParts.append(
        "\n\n--- IMPORTANT REMINDERS ---\n"
        "1. Role: AI assistant, NOT licensed therapist. No diagnoses/treatment.\n"
        "2. Ethics: Adhere to initial system prompt guidelines.\n"
        "3. Crisis: If user mentions self-harm/suicide/abuse, respond empathetically, "
        "urge professional help, provide crisis resources (e.g., 988 US). Do not manage crisis.\n"
        "4. Context: Refer to User Background, Therapy Notes, and Past Conversations.\n"
        "--- END IMPORTANT REMINDERS ---"
    )

    combinedSystemInstruction = "".join(systemInstructionParts)

    # --- Log Initial Context to Pre-Log File ---
    if preLogFile:
        preLogFile.write("\n--- System Instruction Sent to Model ---\n")
        preLogFile.write(combinedSystemInstruction)
        preLogFile.write("\n--- End System Instruction ---\n\n")
        preLogFile.flush()

    console.print(Rule("Initializing AI Model", style="dim"))

    # --- Initialize Google AI ---

    try:

        genai.configure(api_key=API_KEY)

        # Create the model with the combined system instruction
        model = genai.GenerativeModel(
            MODEL_NAME,
            safety_settings=SAFETY_SETTINGS,
            system_instruction=combinedSystemInstruction,
        )

        # Start a chat session (history is managed internally by the SDK)
        # The 'history' parameter here is for *this specific session's* turns,
        # not the long-term history we loaded into the system prompt.
        chat = model.start_chat(history=[])

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

        errorMsgContent = f"Fatal Error during AI Initialization: {e}\nCheck API key, model name, and network connection."
        errorMsg = Panel(
            f"[bold red]{errorMsgContent}[/]",
            title="Initialization Error",
            border_style="red",
        )
        console.print(errorMsg)
        if preLogFile:
            preLogFile.write(
                f"\n{errorMsgContent}\n--- SESSION ENDED DUE TO INIT ERROR ---\n"
            )
            preLogFile.close()

        return

    # --- Open Chat Log File ---
    if logDirsCreated and chatLogFilename:
        try:
            chatLogFile = open(chatLogFilename, "w", encoding="utf-8")
            # Use formatted start time for the header
            chatLogFile.write(f"--- Chat Log Start: {formatted_start_time} ---\n")
            chatLogFile.write(f"Model: {MODEL_NAME}\n")
            chatLogFile.flush()
        except IOError as e:
            console.print(
                f"[bold red]Error opening chat log file {chatLogFilename}:[/] {e}"
            )
            chatLogFile = None
            chatLogFilename = None

    # --- Conversation Loop ---
    messageTimestamps = deque()
    i = 0

    response = (
        console.input(
            "Would you like to use the file for the first prompt? ([y]/n): ",
            markup=False,
        )
        .strip()
        .lower()
        or "y"
    )
    useFileFirstPrompt = response.startswith("y")

    try:  # Main loop try block

        while True:
            current_time_str = datetime.datetime.now().strftime(
                LOG_DETAIL_TIMESTAMP_FORMAT
            )

            # Get user input
            try:
                if useFileFirstPrompt and i == 0:
                    userInput = Path("prompt.txt").read_text()
                    console.print(f"\n[bold blue]You: [/bold blue] {userInput}")
                    i += 1
                else:
                    userInput = console.input(
                        Text("\nYou: ", style="bold blue")
                    ).strip()

                if not userInput:
                    continue

            except EOFError:
                console.print(
                    "\n[yellow]AI Therapist: Input stream closed. Ending session.[/]"
                )
                if chatLogFile:
                    chatLogFile.write("\n--- Input stream closed by user ---\n")
                break
            except KeyboardInterrupt:
                console.print(
                    "\n\n[yellow]AI Therapist: Session interrupted by user. Goodbye.[/]"
                )
                if chatLogFile:
                    chatLogFile.write(
                        "\n--- Session interrupted by user (KeyboardInterrupt during input) ---\n"
                    )
                break

            # Check for exit command
            if userInput.lower() in ["quit", "exit"]:
                console.print(
                    "\n[yellow]AI Therapist: Ending session as requested. Take care.[/]"
                )
                if chatLogFile:
                    chatLogFile.write(
                        "\n--- Session ended by user command ('quit'/'exit') ---\n"
                    )
                break

            # --- Log User Input to Chat Log ---
            if chatLogFile:
                # Use the new timestamp format
                chatLogFile.write(f"\n[{current_time_str}] You: {userInput}\n")
                chatLogFile.flush()

            # --- Rate Limiting Check ---
            currentTime = time.time()
            while messageTimestamps and currentTime - messageTimestamps[0] > 60:
                messageTimestamps.popleft()

            if len(messageTimestamps) >= REQUESTS_PER_MINUTE:
                timeSinceOldest = currentTime - messageTimestamps[0]
                waitTime = 60 - timeSinceOldest + 0.1
                waitMsg = (
                    f"Rate limit reached ({REQUESTS_PER_MINUTE}/min). Please wait..."
                )
                console.print(
                    Panel(
                        f"[bold yellow]{waitMsg}[/]",
                        border_style="yellow",
                        expand=False,
                    )
                )
                if chatLogFile:
                    # Use the new timestamp format for system messages too
                    system_time_str = datetime.datetime.now().strftime(
                        LOG_DETAIL_TIMESTAMP_FORMAT
                    )
                    chatLogFile.write(
                        f"\n[{system_time_str}] System: {waitMsg.strip()}\n"
                    )

                with Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(bar_width=None),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console,
                    transient=True,
                ) as progress:
                    waitTask = progress.add_task(
                        "[yellow]Time until next message:[/]", total=waitTime
                    )
                    while not progress.finished:
                        progress.update(waitTask, advance=0.1)
                        time.sleep(0.1)

                if chatLogFile:
                    chatLogFile.write(f"  (Rate limit wait finished)\n")

                currentTime = time.time()
                while messageTimestamps and currentTime - messageTimestamps[0] > 60:
                    messageTimestamps.popleft()

            # --- Send Message and Get Response ---
            console.print(Text("\nAI Therapist: (Thinking...)", style="italic dim"))
            aiResponseText = None

            try:
                messageTimestamps.append(time.time())
                response = chat.send_message(userInput)
                response_time_str = datetime.datetime.now().strftime(
                    LOG_DETAIL_TIMESTAMP_FORMAT
                )  # Get time after response

                # --- Process Response ---
                if not response.parts:
                    blockReason = "Unknown"
                    if (
                        hasattr(response, "prompt_feedback")
                        and response.prompt_feedback.block_reason
                    ):
                        blockReason = response.prompt_feedback.block_reason.name
                    elif (
                        hasattr(response, "candidates")
                        and response.candidates
                        and response.candidates[0].finish_reason.name != "STOP"
                    ):
                        blockReason = f"Finish Reason: {response.candidates[0].finish_reason.name}"

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
                    if chatLogFile:
                        # Use the new timestamp format
                        chatLogFile.write(
                            f"\n[{response_time_str}] System: {warningMsgContent}\n"
                        )
                else:
                    aiResponseText = response.text

                console.print(
                    Panel(
                        Markdown(aiResponseText),
                        title="AI Therapist",
                        border_style="cyan",
                        expand=False,
                    )
                )

                # --- Log AI Response to Chat Log ---
                if chatLogFile:
                    # Use the new timestamp format
                    chatLogFile.write(
                        f"\n[{response_time_str}] AI Therapist: {aiResponseText}\n"
                    )
                    chatLogFile.flush()

            # --- API Error Handling within the loop ---
            except google.api_core.exceptions.PermissionDenied as e:
                error_time_str = datetime.datetime.now().strftime(
                    LOG_DETAIL_TIMESTAMP_FORMAT
                )
                errorMsgContent = f"API Error: Permission Denied. Check API key/model access. Details: {e}"
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]",
                    title="API Error",
                    border_style="red",
                )
                console.print(errorMsg)
                if chatLogFile:
                    chatLogFile.write(
                        f"\n[{error_time_str}] System Error: {errorMsgContent}\n"
                    )
                # break # Optional: break on permission denied

            except google.api_core.exceptions.NotFound as e:
                error_time_str = datetime.datetime.now().strftime(
                    LOG_DETAIL_TIMESTAMP_FORMAT
                )
                errorMsgContent = f"API Error: Model '{MODEL_NAME}' Not Found or unavailable. Details: {e}"
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]",
                    title="API Error",
                    border_style="red",
                )
                console.print(errorMsg)
                if chatLogFile:
                    chatLogFile.write(
                        f"\n[{error_time_str}] System Error: {errorMsgContent}\n--- SESSION ENDED DUE TO API ERROR ---\n"
                    )
                break

            except google.api_core.exceptions.ResourceExhausted as e:
                error_time_str = datetime.datetime.now().strftime(
                    LOG_DETAIL_TIMESTAMP_FORMAT
                )
                errorMsgContent = f"API Error: Quota Exceeded. Details: {e}"
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]",
                    title="API Error",
                    border_style="red",
                )
                console.print(errorMsg)
                if messageTimestamps:
                    messageTimestamps.pop()
                if chatLogFile:
                    chatLogFile.write(
                        f"\n[{error_time_str}] System Error: {errorMsgContent}\n--- SESSION ENDED DUE TO API ERROR ---\n"
                    )
                break

            except google.api_core.exceptions.InvalidArgument as e:
                error_time_str = datetime.datetime.now().strftime(
                    LOG_DETAIL_TIMESTAMP_FORMAT
                )
                errorMsgContent = (
                    f"API Error: Invalid Argument (check prompt/content). Details: {e}"
                )
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]",
                    title="API Error",
                    border_style="red",
                )
                console.print(errorMsg)
                if messageTimestamps:
                    messageTimestamps.pop()
                if chatLogFile:
                    chatLogFile.write(
                        f"\n[{error_time_str}] System Error: {errorMsgContent}\n"
                    )
                    # Optionally log userInput here if needed for debugging

            except Exception as e:
                error_time_str = datetime.datetime.now().strftime(
                    LOG_DETAIL_TIMESTAMP_FORMAT
                )
                errorMsgContent = f"Error during AI response generation: {e}"
                errorMsg = Panel(
                    f"[bold red]{errorMsgContent}[/]", title="Error", border_style="red"
                )
                console.print(errorMsg)
                if messageTimestamps:
                    messageTimestamps.pop()
                if chatLogFile:
                    chatLogFile.write(
                        f"\n[{error_time_str}] System Error: {errorMsgContent}\n"
                    )

    # --- General Error Handling & Cleanup ---
    except Exception as e:
        error_time_str = datetime.datetime.now().strftime(LOG_DETAIL_TIMESTAMP_FORMAT)
        errorMsgContent = f"An unexpected error occurred in the main loop: {e}"
        errorMsg = Panel(
            f"[bold red]{errorMsgContent}[/]",
            title="Unexpected Error",
            border_style="red",
        )
        console.print(errorMsg)
        if chatLogFile:
            chatLogFile.write(
                f"\n[{error_time_str}] System Error: {errorMsgContent}\n--- SESSION ENDED DUE TO UNEXPECTED ERROR ---\n"
            )

    finally:
        end_time_str = datetime.datetime.now().strftime(LOG_FILE_TIMESTAMP_FORMAT)
        # Ensure the log files are closed properly
        if preLogFile:
            preLogFile.write(f"\n--- Pre-Chat Log End: {end_time_str} ---\n")
            preLogFile.close()
            if preLogFilename:
                console.print(f"Pre-chat log saved to: [cyan]{preLogFilename}[/]")

        if chatLogFile:
            chatLogFile.write(f"\n--- Session End: {end_time_str} ---\n")
            chatLogFile.close()
            if chatLogFilename:
                console.print(f"Chat log saved to: [cyan]{chatLogFilename}[/]")

        console.print(Rule("[bold cyan]AI Therapist Session Ended[/]", style="cyan"))


# --- Main Execution ---
if __name__ == "__main__":

    main()
