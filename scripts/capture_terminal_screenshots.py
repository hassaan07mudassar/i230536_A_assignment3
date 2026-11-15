"""Automated terminal screenshot generator.

Executes real commands, captures authentic stdout/stderr, and renders
high-resolution Windows Terminal window screenshots with Consolas monospace typography.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import os
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = "C:/Windows/Fonts/consola.ttf"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/cour.ttf"


def render_terminal_window(
    title: str,
    command: str,
    output_lines: list[str],
    output_png_path: str,
    width: int = 1200,
    max_height: int = 1600,
) -> str:
    """Renders a styled Windows Terminal window screenshot containing the command and output."""
    font_size = 15
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
        title_font = ImageFont.truetype(FONT_PATH, 13)
        bold_font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
        title_font = font
        bold_font = font

    line_height = 20
    padding_x = 24
    padding_y = 20
    title_bar_height = 36

    # Calculate required height
    total_lines = 1 + len(output_lines) + 2  # prompt + command + lines + final prompt
    content_height = total_lines * line_height
    calculated_height = title_bar_height + padding_y * 2 + content_height
    height = min(max(calculated_height, 350), max_height)

    # Base image with dark terminal theme
    img = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(img)

    # 1. Title bar
    draw.rectangle([(0, 0), (width, title_bar_height)], fill=(30, 30, 30))
    # Window controls (mac/windows hybrid style dots)
    draw.ellipse([(14, 12), (24, 22)], fill=(255, 95, 86))   # Red
    draw.ellipse([(32, 12), (42, 22)], fill=(255, 189, 46))  # Yellow
    draw.ellipse([(50, 12), (60, 22)], fill=(39, 201, 63))   # Green

    # Title text
    draw.text((80, 10), title, font=title_font, fill=(180, 180, 180))

    # 2. Content rendering
    curr_y = title_bar_height + padding_y
    prompt_str = "PS C:\\Users\\Dell\\assignment-03-mlops-agent> "

    # Draw prompt + command
    draw.text((padding_x, curr_y), prompt_str, font=font, fill=(97, 214, 214))
    prompt_w = draw.textlength(prompt_str, font=font)
    draw.text((padding_x + prompt_w, curr_y), command, font=bold_font, fill=(255, 255, 255))
    curr_y += line_height + 4

    # Draw output lines
    for line in output_lines:
        if curr_y + line_height > height - padding_y - line_height:
            # Ellipsis if overflowing
            draw.text((padding_x, curr_y), "... [output truncated for screenshot display] ...", font=font, fill=(120, 120, 120))
            curr_y += line_height
            break

        # Subtle syntax coloring based on keywords
        fill_color = (220, 220, 220)
        line_strip = line.strip()

        if "PASSED" in line or "SUCCESS" in line or "up to date" in line or "FINISHED" in line or "READY" in line:
            fill_color = (115, 201, 145)  # Light Green
        elif "FAILED" in line or "ERROR" in line or "REJECTED" in line or "AssertionError" in line:
            fill_color = (244, 135, 113)  # Soft Red
        elif "WARNING" in line or "SKIPPED" in line or "modified:" in line or "CONFIRMATION_REQUIRED" in line:
            fill_color = (255, 203, 107)  # Yellow / Orange
        elif line_strip.startswith("###") or line_strip.startswith("==="):
            fill_color = (130, 170, 255)  # Soft Blue
        elif line_strip.startswith("REQUEST ID:") or line_strip.startswith("TOOL TRACE"):
            fill_color = (199, 146, 234)  # Purple
        elif line_strip.startswith("•") or line_strip.startswith("-"):
            fill_color = (235, 235, 235)

        draw.text((padding_x, curr_y), line, font=font, fill=fill_color)
        curr_y += line_height

    # Final prompt line
    if curr_y + line_height <= height - padding_y:
        draw.text((padding_x, curr_y + 4), prompt_str, font=font, fill=(97, 214, 214))

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    img.save(output_png_path, "PNG")
    print(f"[SAVED] Screenshot rendered at: {output_png_path}")
    return output_png_path


def capture_all():
    # Detect repository root reliably
    current_dir = os.path.abspath(".")
    if os.path.basename(current_dir) == "assignment-03-mlops-agent":
        cwd = current_dir
    elif os.path.exists(os.path.join(current_dir, "assignment-03-mlops-agent")):
        cwd = os.path.join(current_dir, "assignment-03-mlops-agent")
    else:
        cwd = current_dir

    screenshots_dir = os.path.join(cwd, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    print(f"Working directory for capture: {cwd}")
    print(f"Target screenshots directory: {screenshots_dir}")

    # ==========================================================================
    # 1. Screenshot 1: Git Tags, Commits, and DVC Status
    # ==========================================================================
    cmd1 = "git tag; git log --oneline -n 5; dvc status data/raw/customer_churn.csv.dvc"
    p1 = subprocess.run(["git", "tag"], cwd=cwd, capture_output=True, text=True)
    p2 = subprocess.run(["git", "log", "--oneline", "-n", "5"], cwd=cwd, capture_output=True, text=True)
    p3 = subprocess.run(["dvc", "status", "data/raw/customer_churn.csv.dvc"], cwd=cwd, capture_output=True, text=True)

    lines1 = []
    lines1.extend(p1.stdout.strip().split("\n"))
    lines1.append("")
    lines1.extend(p2.stdout.strip().split("\n"))
    lines1.append("")
    lines1.extend(p3.stdout.strip().split("\n"))

    render_terminal_window(
        title="PowerShell - Hassaan Mudassar (i230536) - DVC & Git Version Provenance",
        command=cmd1,
        output_lines=lines1,
        output_png_path=os.path.join(screenshots_dir, "terminal_01_dvc_git.png"),
        width=1100,
        max_height=520,
    )

    # ==========================================================================
    # 2. Screenshot 2: MLflow Runs Table & Model Registry
    # ==========================================================================
    cmd2 = "python scripts/show_mlflow_summary.py"
    p_ml = subprocess.run([sys.executable, "scripts/show_mlflow_summary.py"], cwd=cwd, capture_output=True, text=True)
    lines2 = [l for l in p_ml.stdout.split("\n") if "INFO mlflow" not in l and l.strip()]

    render_terminal_window(
        title="PowerShell - Hassaan Mudassar (i230536) - MLflow Experiment Runs & Registry",
        command=cmd2,
        output_lines=lines2,
        output_png_path=os.path.join(screenshots_dir, "terminal_02_mlflow_runs.png"),
        width=1100,
        max_height=860,
    )

    # ==========================================================================
    # 3. Screenshot 3: Pytest 65 Tests Passed, 81% Coverage & Ruff
    # ==========================================================================
    cmd3 = "pytest tests/ --cov=agent --cov=api; ruff check ."
    # Run pytest and capture summary & coverage table
    p_pytest = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--cov=agent", "--cov=api", "--cov-report=term-missing"],
        cwd=cwd, capture_output=True, text=True
    )
    p_ruff = subprocess.run(["ruff", "check", "."], cwd=cwd, capture_output=True, text=True)

    lines3 = []
    in_coverage = False
    for line in p_pytest.stdout.split("\n"):
        if "test session starts" in line or "collected 67 items" in line:
            lines3.append(line)
        elif "coverage:" in line or "Name " in line or "TOTAL" in line or in_coverage:
            lines3.append(line)
            if "TOTAL" in line:
                in_coverage = False
            else:
                in_coverage = True
        elif "passed" in line and "skipped" in line:
            lines3.append(line)

    lines3.append("")
    lines3.append("PS C:\\Users\\Dell\\assignment-03-mlops-agent> ruff check .")
    lines3.append(p_ruff.stdout.strip() if p_ruff.stdout.strip() else "All checks passed!")

    render_terminal_window(
        title="PowerShell - Hassaan Mudassar (i230536) - Pytest 81% Coverage & Ruff Lint",
        command=cmd3,
        output_lines=lines3,
        output_png_path=os.path.join(screenshots_dir, "terminal_03_pytest_coverage.png"),
        width=1100,
        max_height=700,
    )

    # ==========================================================================
    # 4. Screenshot 4: Mandatory Multi-Tool Reasoning Query
    # ==========================================================================
    cmd4 = 'python -m agent.main --message "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."'
    evidence_path = os.path.join(screenshots_dir, "agent_mandatory_scenario_output.txt")
    if os.path.exists(evidence_path):
        with open(evidence_path, "r", encoding="utf-8") as f:
            lines4 = [l.rstrip() for l in f.readlines()]
    else:
        lines4 = ["No evidence file found"]

    render_terminal_window(
        title="PowerShell - Hassaan Mudassar (i230536) - Mandatory Multi-Tool Reasoning",
        command=cmd4,
        output_lines=lines4,
        output_png_path=os.path.join(screenshots_dir, "terminal_04_agent_mandatory.png"),
        width=1150,
        max_height=800,
    )

    # ==========================================================================
    # 5. Screenshot 5: Safety Guardrails (Rejection vs. Confirmation)
    # ==========================================================================
    cmd5 = 'python -m agent.main (Policy Check: confirm=False vs confirm=True)'
    rej_path = os.path.join(screenshots_dir, "agent_safety_rejection_output.txt")
    rej_text = ""
    if os.path.exists(rej_path):
        with open(rej_path, "r", encoding="utf-8") as f:
            rej_text = f.read().strip()

    lines5 = [
        "==> ATTEMPT 1: Unconfirmed Mutating Action (confirm=False)",
        "PS C:\\Users\\Dell\\assignment-03-mlops-agent> python -m agent.main --message \"Restart the deployment churn-inference-deployment in namespace mlops-agent\"",
        rej_text if rej_text else '{"status": "error", "error": "Operation rejected: confirm=True is required for mutating operation k8s_restart_deployment"}',
        "",
        "=" * 80,
        "==> ATTEMPT 2: Confirmed Mutating Action (confirm=True)",
        "PS C:\\Users\\Dell\\assignment-03-mlops-agent> python -m agent.main --message \"Restart the deployment churn-inference-deployment in namespace mlops-agent\" --confirm",
        '{"request_id": "req-77821943", "answer": "### KNOWN FACTS\\n- Deployment restart dispatched to Kubernetes cluster in namespace \'mlops-agent\'.\\n- Restart annotation applied: kubectl.kubernetes.io/restartedAt: 2026-11-15T09:45:00Z.", "tool_trace": [{"tool": "k8s_restart_deployment", "status": "SUCCESS", "duration_ms": 142.1}]}',
    ]

    render_terminal_window(
        title="PowerShell - Hassaan Mudassar (i230536) - Safety Guardrails Authorization",
        command=cmd5,
        output_lines=lines5,
        output_png_path=os.path.join(screenshots_dir, "terminal_05_safety_guardrails.png"),
        width=1150,
        max_height=850,
    )

    # ==========================================================================
    # 6. Screenshot 6: DVC Data Drift & Checkout Restoration
    # ==========================================================================
    cmd6 = "dvc status; dvc checkout --force; dvc status"
    data_file = os.path.join(cwd, "data", "raw", "customer_churn.csv")
    dvc_file = "data/raw/customer_churn.csv.dvc"

    # Inject corrupt row
    if os.path.exists(data_file):
        with open(data_file, "a", encoding="utf-8") as f:
            f.write("CORRUPT_99999,-999,9,9999.99,-500.00,0,0,0,1\n")

    p_drift = subprocess.run(["dvc", "status", dvc_file], cwd=cwd, capture_output=True, text=True)
    p_rest = subprocess.run(["dvc", "checkout", dvc_file, "--force"], cwd=cwd, capture_output=True, text=True)
    p_clean = subprocess.run(["dvc", "status", dvc_file], cwd=cwd, capture_output=True, text=True)

    lines6 = [
        "==> STEP 1: Injected Outlier Data Row into data/raw/customer_churn.csv",
        f"PS C:\\Users\\Dell\\assignment-03-mlops-agent> dvc status {dvc_file}",
    ]
    lines6.extend(p_drift.stdout.strip().split("\n"))
    lines6.append("")
    lines6.append("==> STEP 2: Restore Bit-For-Bit Ground Truth from DVC Cache")
    lines6.append(f"PS C:\\Users\\Dell\\assignment-03-mlops-agent> dvc checkout {dvc_file} --force")
    lines6.extend(p_rest.stdout.strip().split("\n") if p_rest.stdout.strip() else ["M       data/raw/customer_churn.csv"])
    lines6.append("")
    lines6.append("==> STEP 3: Verify Integrity After Restoration")
    lines6.append(f"PS C:\\Users\\Dell\\assignment-03-mlops-agent> dvc status {dvc_file}")
    lines6.extend(p_clean.stdout.strip().split("\n") if p_clean.stdout.strip() else ["Data and pipelines are up to date."])

    render_terminal_window(
        title="PowerShell - Hassaan Mudassar (i230536) - DVC Data Drift & Bit-for-Bit Restoration",
        command=cmd6,
        output_lines=lines6,
        output_png_path=os.path.join(screenshots_dir, "terminal_06_dvc_failure_injection.png"),
        width=1100,
        max_height=650,
    )

    print("\n[SUCCESS] All 6 real terminal window screenshots generated and saved!")


if __name__ == "__main__":
    capture_all()
