from collections import OrderedDict
from pathlib import Path

from natsort import natsorted
from PyPDF2 import PdfReader
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table

console = Console()

NOTES_DIR = Path("/Users/kadengruizenga/Documents/Other/Notes")


def ExtractTextFromPdf(pdfPath: Path) -> str:
    """
    Extracts text from a PDF file.

    Parameters
    ----------
    pdfPath : Path
        The path to the PDF file.

    Returns
    -------
    str
        The extracted text from the PDF.
    """
    reader = PdfReader(pdfPath)
    text = ""

    for page in reader.pages:

        text += page.extract_text() or ""

    return text


if __name__ == "__main__":

    notesFiles = natsorted(list(NOTES_DIR.iterdir()))

    fileContent = OrderedDict()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}", justify="left"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        expand=True,
    ) as progress:

        task = progress.add_task("Extracting text from PDFs...", total=len(notesFiles))

        for file in notesFiles:

            progress.update(
                task, description=f"Extracting text from {file.name}", refresh=True
            )

            fileContent[file.name] = ExtractTextFromPdf(file)

            progress.update(
                task,
                description=f"Extracted text from {file.name}",
                refresh=True,
                advance=1,
            )


a = ""
for key, val in fileContent.items():

    a += f"{key}\n\n{val}\n\n--------------------------------------\n\n"

Path("out.txt").write_text(a)
