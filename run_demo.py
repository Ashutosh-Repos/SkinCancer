import os
import sys
import subprocess
import time

# --- Auto-Switch to Venv ---
VENV_PYTHON = os.path.join(os.getcwd(), '.venv', 'bin', 'python3')
if os.path.exists(VENV_PYTHON) and sys.executable != VENV_PYTHON:
    # Re-run the script using the virtual environment python
    os.execv(VENV_PYTHON, [VENV_PYTHON] + sys.argv)

# --- Rich Fallback Support ---
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.spinner import Spinner
    from rich import print as rprint
    USE_RICH = True
except ImportError:
    USE_RICH = False

class SimpleConsole:
    def print(self, msg, *args, **kwargs): print(msg)
    def status(self, msg):
        class DummyStatus:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyStatus()

if USE_RICH:
    console = Console()
else:
    console = SimpleConsole()
    def Panel(msg, **kwargs): return msg
    def Table(**kwargs):
        class DummyTable:
            def add_column(self, *args, **kwargs): pass
            def add_row(self, *args, **kwargs): pass
        return DummyTable()

# Configuration
VENV_PYTHON = os.path.join(os.getcwd(), '.venv', 'bin', 'python3')
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = 'python3'  # Fallback

DEFAULT_MODEL = "models/checkpoints/sequential_best.h5"
SAMPLE_IMAGE = "data/images/ISIC_0024306.jpg"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def show_header():
    console.print(Panel.fit(
        "[bold cyan]🔬 Skin Cancer Detection - Demo Launcher[/bold cyan]\n"
        "[dim]Project finalize & Running State Verification[/dim]",
        border_style="cyan"
    ))

def run_verify():
    with console.status("[bold green]Running System Diagnostics...") as status:
        time.sleep(1)
        # Check files
        files_to_check = [
            DEFAULT_MODEL,
            "data/HAM10000_metadata.csv",
            "data/norm_stats.json"
        ]
        
        table = Table(title="File Integrity Check")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="bold")
        
        for f in files_to_check:
            exists = os.path.exists(f)
            status_text = "[green]OK[/green]" if exists else "[red]MISSING[/red]"
            table.add_row(f, status_text)
        
        console.print(table)
        
        # Check python env
        try:
            version = subprocess.check_output([VENV_PYTHON, "--version"]).decode().strip()
            console.print(f"\n[green]✅ Python Environment:[/green] {version}")
        except Exception:
            console.print("\n[red]❌ Virtual environment not detected or broken.[/red]")
            
    input("\nPress Enter to return to menu...")

def run_inference():
    clear_screen()
    show_header()
    console.print(f"[bold yellow]🖼️  Running Inference on Sample Image[/bold yellow]")
    console.print(f"[dim]Image: {SAMPLE_IMAGE}[/dim]\n")
    
    cmd = [
        VENV_PYTHON, "src/inference.py",
        "--model", DEFAULT_MODEL,
        "--image", SAMPLE_IMAGE,
        "--top-k", "3"
    ]
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        console.print(f"[red]Error running inference: {e}[/red]")
        
    input("\nPress Enter to return to menu...")

def run_camera():
    clear_screen()
    show_header()
    console.print("[bold green]📸 Launching Real-time Camera Service[/bold green]")
    console.print("[yellow]IMPORTANT: Click on the camera window to focus it first.[/yellow]")
    console.print("[dim]Press 'q' in the window to quit, 's' to save screenshot.[/dim]\n")
    
    cmd = [
        VENV_PYTHON, "src/camera_service.py",
        "--model", DEFAULT_MODEL,
        "--mode", "camera"
    ]
    
    try:
        subprocess.run(cmd)
    except Exception as e:
        console.print(f"[red]Error running camera: {e}[/red]")
        input("\nPress Enter to return to menu...")

def run_api():
    clear_screen()
    show_header()
    console.print("[bold magenta]🌐 Starting REST API Service & Web UI[/bold magenta]")
    console.print("[green]Diagnostic UI active at: http://localhost:5005/[/green]")
    console.print("[dim]Backend Health: http://localhost:5005/health[/dim]\n")
    console.print("[yellow]Note: Close this terminal or press Ctrl+C to stop the server.[/yellow]\n")
    
    cmd = [
        VENV_PYTHON, "src/camera_service.py",
        "--model", DEFAULT_MODEL,
        "--mode", "api",
        "--port", "5005"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[yellow]API Service stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error starting API: {e}[/red]")
        input("\nPress Enter to return to menu...")


def run_comparison():
    clear_screen()
    show_header()
    console.print("[bold yellow]📊 Model Performance Comparison Report[/bold yellow]\n")
    
    csv_path = "results/model_comparison.csv"
    if not os.path.exists(csv_path):
        console.print("[red]Comparison data not found. Please run evaluation on multiple models first.[/red]")
    else:
        import csv
        table = Table(title="Academic Model Comparison")
        try:
            with open(csv_path, mode='r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for h in header:
                    table.add_column(h, style="cyan")
                
                for row in reader:
                    table.add_row(*row)
            console.print(table)
        except Exception as e:
            console.print(f"[red]Error reading comparison data: {e}[/red]")
            
    input("\nPress Enter to return to menu...")

def main_menu():
    while True:
        clear_screen()
        show_header()
        
        console.print("Please select a demo mode:")
        console.print("1. [bold cyan]🚀 Perform Inference Verification[/bold cyan] (Single Image)")
        console.print("2. [bold green]📸 Start Real-time Camera Demo[/bold green]")
        console.print("3. [bold magenta]🌐 Start REST API Server[/bold magenta]")
        console.print("4. [bold yellow]📊 View performance Comparison[/bold yellow]")
        console.print("5. [bold yellow]⚙️  Run System Diagnostics[/bold yellow]")
        console.print("6. [bold red]❌ Exit[/bold red]")
        
        choice = input("\nSelect an option (1-6): ")
        
        if choice == '1':
            run_inference()
        elif choice == '2':
            run_camera()
        elif choice == '3':
            run_api()
        elif choice == '4':
            run_comparison()
        elif choice == '5':
            run_verify()
        elif choice == '6':
            console.print("\n[bold cyan]Goodbye![/bold cyan]")
            break
        else:
            console.print("[red]Invalid choice. Try again.[/red]")
            time.sleep(1)

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Goodbye![/bold cyan]")
        sys.exit(0)
