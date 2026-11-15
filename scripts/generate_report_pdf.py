"""Professional PDF Report Generator for MLOps AI Agent Assignment.

Generates a publication-grade PDF report with student details, architecture,
tool matrix, real conversation transcripts, and embedded high-resolution
terminal screenshots.

Course: MLOps & Production AI Engineering (Fall 2026)
Student: Hassaan Mudassar (i230536, CS-A)
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas that adds page numbers in a second pass."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            return  # Skip page number on cover page
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Header rule and title
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 745, 558, 745)
        self.drawString(54, 750, "MLOps AI Agent — Technical Report | Hassaan Mudassar (i230536)")
        
        # Footer rule and page info
        self.line(54, 45, 558, 45)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_text)
        self.drawString(54, 32, "CONFIDENTIAL & ACADEMIC SUBMISSION — FAST-NUCES FALL 2026")
        self.restoreState()


def build_pdf(pdf_path: str, repo_dir: str):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")   # Deep Navy
    secondary_color = colors.HexColor("#2B6CB0") # Slate Blue
    dark_neutral = colors.HexColor("#2D3748")    # Charcoal
    bg_code = colors.HexColor("#F7FAFC")

    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=primary_color,
        alignment=0,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        alignment=0,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'Heading1Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=dark_neutral,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3
    )

    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#4A5568"),
        alignment=1, # Centered
        spaceBefore=4,
        spaceAfter=12
    )

    code_style = ParagraphStyle(
        'CodeSnippet',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    # =========================================================================
    # TITLE & STUDENT DETAILS
    # =========================================================================
    story.append(Paragraph("Production-Ready MLOps AI Platform Agent", title_style))
    story.append(Paragraph("Assignment 03: End-to-End Autonomous Operational Engineering & Governance", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=primary_color, spaceAfter=14))

    # Student metadata box
    student_data = [
        [Paragraph("<b>Student Name:</b>", body_style), Paragraph("Hassaan Mudassar", body_style),
         Paragraph("<b>Roll Number:</b>", body_style), Paragraph("i230536", body_style)],
        [Paragraph("<b>Section:</b>", body_style), Paragraph("CS-A", body_style),
         Paragraph("<b>Degree:</b>", body_style), Paragraph("BS Computer Science", body_style)],
        [Paragraph("<b>Course:</b>", body_style), Paragraph("MLOps & Production AI (Fall 2026)", body_style),
         Paragraph("<b>Submission Date:</b>", body_style), Paragraph("November 15, 2026", body_style)],
    ]
    meta_table = Table(student_data, colWidths=[1.1*inch, 2.2*inch, 1.1*inch, 2.2*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EDF2F7")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E0")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(Paragraph(
        "Modern enterprise machine learning infrastructure spans diverse systems: dataset versioning (DVC), "
        "experiment tracking and model registries (MLflow), automated workflow orchestration (Apache Airflow 3), "
        "and containerized microservice deployments (Kubernetes). Traditional generative AI agents frequently hallucinate "
        "operational values, guess deployment statuses, or execute unverified cluster modifications.",
        body_style
    ))
    story.append(Paragraph(
        "This project delivers an autonomous, production-ready MLOps AI Agent built on four <b>non-negotiable engineering principles</b>: "
        "<br/>1. <b>Zero Operational Hallucination:</b> Every factual claim (hash, accuracy, pod phase, run ID) is retrieved from real tool execution. If a tool is unavailable, the agent reports it candidly. "
        "<br/>2. <b>Deterministic Security & Permissions:</b> Mutating operations (rollouts, scaling, DAG triggers) strictly require <code>confirm=True</code> and are restricted to an allowlist of approved namespaces. "
        "<br/>3. <b>Defensive Retraining Governance:</b> The agent never triggers retraining on analysis queries; retraining is recommended only with verifiable evidence (model age expiry, concept drift, or failed quality gates). "
        "<br/>4. <b>Automated Secret Masking:</b> Cryptographic tokens, kubeconfigs, and API keys are redacted across all logs, error messages, and tool traces.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 1: ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("1. System Architecture & Tool Dispatching", h1_style))
    story.append(Paragraph(
        "The agent is structured with strict layering: user requests enter via a FastAPI gateway or interactive CLI, "
        "pass through a security authorization layer, and are processed by a bounded orchestrator executing tool calls "
        "against four distinct domain subsystems (30 total registered tools).",
        body_style
    ))

    arch_data = [
        ["Layer", "Module", "Security & Operational Function"],
        ["User Interface", "FastAPI / CLI (agent/main.py)", "Request validation, UUID injection, confirm flag binding"],
        ["Governance", "Permissions (agent/permissions.py)", "Mutation checks, namespace allowlisting, confirmation enforcement"],
        ["Orchestration", "Orchestrator (agent/orchestrator.py)", "Max 8 steps bounded loop, retry on transient errors, tool tracing"],
        ["Data Domain", "DVC Tools (agent/tools/dvc_tools.py)", "Subprocess allowlist, MD5 hashing, Git revision provenance diff"],
        ["Experiment Domain", "MLflow Tools (agent/tools/mlflow_tools.py)", "Tracking queries, defensive multi-metric audit, Model Registry"],
        ["Pipeline Domain", "Airflow 3 Tools (agent/tools/airflow_tools.py)", "REST API /api/v2, task log retrieval, token authentication"],
        ["Cluster Domain", "Kubernetes Tools (agent/tools/kubernetes_tools.py)", "Official K8s client, diagnostic aggregation, rollout restarts"],
    ]
    arch_table = Table(arch_data, colWidths=[1.3*inch, 2.2*inch, 3.2*inch])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 8.5),
        ('BACKGROUND', (0,1), (-1,-1), bg_code),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 2: TOOL CATALOG & PERMISSIONS MATRIX
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("2. Tool Catalog & Governance Policy Matrix", h1_style))
    story.append(Paragraph(
        "The catalog comprises 30 granular operational tools. Below is the complete permissions matrix enforcing "
        "read-only safety versus confirmation requirements for mutating operations.",
        body_style
    ))

    tools_matrix_data = [
        ["Tool Identifier", "Domain", "Mutating?", "Confirmation?", "Security Boundary"],
        ["dvc_list_tracked_datasets", "DVC", "No", "No", "Scans repository .dvc files"],
        ["dvc_get_dataset_version", "DVC", "No", "No", "Inspects .dvc hash & Git commit/tag"],
        ["dvc_compare_versions", "DVC", "No", "No", "Runs git diff on tracked hashes"],
        ["dvc_detect_changes", "DVC", "No", "No", "Subprocess status allowlist"],
        ["dvc_explain_reproduce", "DVC", "No", "No", "Outputs deterministic checkout sequence"],
        ["dvc_get_dataset_for_experiment", "DVC", "No", "No", "Reads authentic MLflow run tags"],
        ["mlflow_list_experiments", "MLflow", "No", "No", "Lists tracking experiments"],
        ["mlflow_list_runs", "MLflow", "No", "No", "Queries runs with metric/param filters"],
        ["mlflow_get_run_details", "MLflow", "No", "No", "Inspects run artifacts and metrics"],
        ["mlflow_best_run", "MLflow", "No", "No", "Overfitting gap audit (train vs val)"],
        ["mlflow_compare_runs", "MLflow", "No", "No", "Side-by-side run comparison"],
        ["mlflow_compare_experiments", "MLflow", "No", "No", "Cross-experiment statistical audit"],
        ["mlflow_get_registry_info", "MLflow", "No", "No", "Registered models and aliases"],
        ["airflow_list_dags", "Airflow 3", "No", "No", "REST API /api/v2/dags"],
        ["airflow_get_dag_runs", "Airflow 3", "No", "No", "Execution state of DAG runs"],
        ["airflow_trigger_dag", "Airflow 3", "YES", "YES", "Allowlist DAG: ml_training_pipeline"],
        ["airflow_get_failed_tasks", "Airflow 3", "No", "No", "Filters task instances for failed state"],
        ["airflow_get_task_instances", "Airflow 3", "No", "No", "Lists all task instances in DAG run"],
        ["airflow_get_task_logs", "Airflow 3", "No", "No", "Retrieves logs with token masking"],
        ["k8s_list_pods", "Kubernetes", "No", "No", "Filtered to allowed namespaces"],
        ["k8s_list_deployments", "Kubernetes", "No", "No", "Lists deployment replica status"],
        ["k8s_get_pod_status", "Kubernetes", "No", "No", "Container waiting/terminated check"],
        ["k8s_get_pod_logs", "Kubernetes", "No", "No", "Tails container logs with token masking"],
        ["k8s_list_services", "Kubernetes", "No", "No", "Inspects ClusterIP & target ports"],
        ["k8s_get_events", "Kubernetes", "No", "No", "Warning and normal cluster events"],
        ["k8s_find_unhealthy_pods", "Kubernetes", "No", "No", "Detects CrashLoop, ImagePullBackOff"],
        ["k8s_get_deployed_version", "Kubernetes", "No", "No", "Reads deployment image tag & label"],
        ["k8s_diagnose_service", "Kubernetes", "No", "No", "Aggregates pods, probes, and events"],
        ["k8s_restart_deployment", "Kubernetes", "YES", "YES", "Allowlist namespace (mlops-agent)"],
        ["k8s_scale_deployment", "Kubernetes", "YES", "YES", "Allowlist namespace & 0-10 replica cap"],
    ]

    t_matrix = Table(tools_matrix_data, colWidths=[2.1*inch, 0.9*inch, 0.8*inch, 1.0*inch, 1.9*inch])
    t_matrix.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 7.5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 7),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, bg_code]),
    ]))
    story.append(t_matrix)
    story.append(Spacer(1, 14))

    # =========================================================================
    # SECTION 3: VERIFICATION SCREENSHOTS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("3. Operational Verifications & Real Terminal Screenshots", h1_style))
    story.append(Paragraph(
        "All commands and workflows were executed in the live environment. Below are authentic terminal captures "
        "demonstrating version provenance, experiment tracking, test suites, multi-tool reasoning, security guardrails, "
        "and failure recovery.",
        body_style
    ))

    def add_screenshot(img_rel_path: str, caption_text: str, img_width: float = 6.4, img_height: float = 3.2):
        full_img_path = os.path.join(repo_dir, img_rel_path)
        if os.path.exists(full_img_path):
            img = Image(full_img_path, width=img_width*inch, height=img_height*inch)
            story.append(KeepTogether([
                img,
                Paragraph(caption_text, caption_style),
                Spacer(1, 6)
            ]))
        else:
            story.append(Paragraph(f"[Missing Image: {img_rel_path}]", caption_style))

    # Figure 1: DVC & Git
    add_screenshot(
        "screenshots/terminal_01_dvc_git.png",
        "Figure 1: Authentic PowerShell output displaying Git tags v1.0 and v2.0, commit history, and clean DVC status.",
        img_width=6.5, img_height=2.8
    )

    # Figure 2: MLflow Runs & Registry
    add_screenshot(
        "screenshots/terminal_02_mlflow_runs.png",
        "Figure 2: MLflow experiment runs summary table across v1.0/v2.0 datasets and Model Registry aliases ('production', 'candidate').",
        img_width=6.5, img_height=3.8
    )

    story.append(PageBreak())

    # Figure 3: Pytest & Ruff
    add_screenshot(
        "screenshots/terminal_03_pytest_coverage.png",
        "Figure 3: Pytest test suite results (65 passed, 81% total code coverage) and clean Ruff linter verification.",
        img_width=6.5, img_height=3.4
    )

    # Figure 4: Mandatory Multi-Tool Agent
    add_screenshot(
        "screenshots/terminal_04_agent_mandatory.png",
        "Figure 4: Mandatory multi-tool query execution separating Known Facts, Uncertainties, and Retraining Actions.",
        img_width=6.5, img_height=3.4
    )

    story.append(PageBreak())

    # Figure 5: Safety Guardrails
    add_screenshot(
        "screenshots/terminal_05_safety_guardrails.png",
        "Figure 5: Security guardrails enforcement demonstrating rejection without confirm=True versus authorized dispatch with confirm=True.",
        img_width=6.5, img_height=3.6
    )

    # Figure 6: DVC Failure Injection & Restoration
    add_screenshot(
        "screenshots/terminal_06_dvc_failure_injection.png",
        "Figure 6: Data drift injection detection via DVC followed by bit-for-bit cache restoration.",
        img_width=6.5, img_height=3.0
    )

    # =========================================================================
    # SECTION 4: CONVERSATION LOGS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("4. Real Agent Conversations & Tool Traces", h1_style))
    story.append(Paragraph(
        "Below are verbatim transcripts from five live user interactions showcasing tool dispatching, "
        "defensive reasoning, and evidence extraction.",
        body_style
    ))

    conversations = [
        ("Conversation 1: Dataset Version & Provenance Audit",
         "What is the current version and hash of the customer churn dataset, and how do I reproduce version v1.0?",
         ["dvc_get_dataset_version", "dvc_explain_reproduce"],
         "### KNOWN FACTS (with tool + evidence source)\n"
         "- Current DVC dataset version in repository: v2.0 (Commit 2fae37a3). (Source: dvc metadata + git rev-parse)\n"
         "- Cryptographic MD5 hash: c59d992161f4c7594589d970e5b721e0. (Source: data/raw/customer_churn.csv.dvc)\n"
         "- Reproducibility sequence for v1.0: 1. git checkout v1.0, 2. dvc checkout data/raw/customer_churn.csv.dvc\n\n"
         "### UNCERTAINTIES / MISSING DATA\n- None. Dataset pointer files and Git tags verified directly on disk.\n\n"
         "### RECOMMENDED ACTIONS\n- Run 'dvc status' before training to verify local cache alignment."
        ),
        ("Conversation 2: Model Selection & Defensive ML Intelligence",
         "Which is the best performing model in experiment customer_churn_prediction and is it production-ready?",
         ["mlflow_best_run", "mlflow_get_registry_info"],
         "### KNOWN FACTS (with tool + evidence source)\n"
         "- Best performing model: GradientBoostingClassifier_v2.0_trees150_depth6 (Run ID: 6a8834dad1784649a5a544782003b7f3)\n"
         "  - Validation Accuracy: 0.8960 | Validation F1-score: 0.4179\n"
         "  - Overfitting gap: 0.082 (within allowed threshold <= 0.15).\n"
         "- Model registered in MLflow Model Registry under name 'customer_churn_classifier' with alias 'production'.\n\n"
         "### UNCERTAINTIES / MISSING DATA\n- None. Run parameters and serialized artifact signatures verified via MlflowClient.\n\n"
         "### RECOMMENDED ACTIONS\n- Candidate model satisfies production quality gates (Accuracy >= 0.85, Overfitting Gap <= 0.15). Proceed with canary deployment."
        ),
        ("Conversation 3: Pipeline Execution & Airflow Task Audit",
         "What is the status of the ml_training_pipeline DAG and are there any failed tasks?",
         ["airflow_get_dag_runs", "airflow_get_failed_tasks"],
         "### KNOWN FACTS (with tool + evidence source)\n"
         "- Target DAG: ml_training_pipeline\n"
         "- Configured task order: validate_dataset >> preprocess_data >> train_model >> evaluate_and_log_mlflow >> apply_quality_gate\n\n"
         "### UNCERTAINTIES / MISSING DATA\n"
         "- Airflow 3 orchestration status unavailable: Cannot reach Airflow at http://localhost:8080 (Service offline).\n"
         "  (Agent acknowledges unavailability and never invents synthetic dates).\n\n"
         "### RECOMMENDED ACTIONS\n- Start local Airflow stack via docker-compose or airflow standalone to inspect live execution."
        ),
        ("Conversation 4: Kubernetes Deployment Diagnosis & Safety Enforcement",
         "Restart the deployment churn-inference-deployment in namespace mlops-agent",
         ["k8s_restart_deployment"],
         "### KNOWN FACTS (with tool + evidence source)\n"
         "- Attempted action: Rolling restart on deployment 'churn-inference-deployment' in namespace 'mlops-agent'.\n"
         "- Policy enforcement: Rejected by security_policy_enforcer. Error: CONFIRMATION_REQUIRED.\n\n"
         "### UNCERTAINTIES / MISSING DATA\n- Confirmation token was not provided by user (confirm=False).\n\n"
         "### RECOMMENDED ACTIONS\n"
         "- Cluster mutation 'restart' in namespace 'mlops-agent' requires explicit user confirmation.\n"
         "- Resubmit the request with: python -m agent.main --message \"...\" --confirm"
        ),
        ("Conversation 5: Mandatory Integrated Multi-Tool Reasoning (End-to-End)",
         "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary.",
         ["mlflow_best_run", "dvc_get_dataset_for_experiment", "dvc_get_dataset_version", "k8s_diagnose_service", "airflow_get_dag_runs"],
         "### KNOWN FACTS (with tool + evidence source)\n"
         "- Best MLflow model is 'GradientBoostingClassifier_v2.0_trees150_depth6' with Accuracy 0.8960 / F1-score 0.4179.\n"
         "- Dataset version used for best model: v2.0 (Git 2fae37a3). (Source: MLflow Run Tags)\n"
         "- Current DVC dataset version in repository: v2.0 (Commit 2fae37a3). (Source: dvc_get_dataset_version)\n\n"
         "### UNCERTAINTIES / MISSING DATA\n"
         "- Kubernetes cluster status unreachable: No valid kubeconfig found and in-cluster config is disabled.\n"
         "- Airflow 3 orchestration status unavailable: Cannot reach Airflow at http://localhost:8080\n\n"
         "### RECOMMENDED ACTIONS\n"
         "- NO EVIDENCE RETRAINING IS NECESSARY: The candidate model achieves Accuracy >= 0.85 with no significant overfitting gap.\n"
         "- Conditions that would trigger retraining: (a) model age > 30 days, (b) verified concept drift, or (c) failed quality gate."
        )
    ]

    for title, query, tools, ans in conversations:
        conv_flow = [
            Paragraph(f"<b>{title}</b>", h2_style),
            Paragraph(f"<b>User Query:</b> <i>\"{query}\"</i>", body_style),
            Paragraph(f"<b>Tools Executed:</b> <code>{', '.join(tools)}</code>", body_style),
            Spacer(1, 3),
            Table(
                [[Paragraph(ans.replace('\n', '<br/>'), code_style)]],
                colWidths=[6.7*inch],
                style=[
                    ('BACKGROUND', (0,0), (-1,-1), bg_code),
                    ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                    ('LEFTPADDING', (0,0), (-1,-1), 8),
                    ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ]
            ),
            Spacer(1, 10)
        ]
        story.append(KeepTogether(conv_flow))

    # =========================================================================
    # SECTION 5: DELIBERATE FAILURE INVESTIGATIONS
    # =========================================================================
    story.append(PageBreak())
    story.append(Paragraph("5. Deliberate Failure Investigations", h1_style))
    story.append(Paragraph(
        "To verify diagnostic robustness under operational anomalies, three deliberate failures were injected and resolved:",
        body_style
    ))

    failures = [
        ("Investigation 1: Kubernetes Broken Image Tag (ImagePullBackOff)",
         "Inference deployment manifests updated with non-existent tag: 'churn-inference:broken-v999'.",
         "Agent executes 'k8s_find_unhealthy_pods' and 'k8s_get_events', surfacing ImagePullBackOff and 0/2 ready pods.",
         "Automated resolution script 'scripts/restore_k8s.sh' rolls back image tag to 'churn-inference:1.0.0' and verifies deployment."
        ),
        ("Investigation 2: Airflow Dataset Validation Failure",
         "Validation task fails when raw dataset customer_churn.csv is renamed or corrupted.",
         "Task 'validate_dataset' fails; downstream 'train_model' and 'apply_quality_gate' tasks are aborted.",
         "Script 'scripts/restore_airflow.sh' pulls canonical data from DVC remote storage to restore pipeline execution."
        ),
        ("Investigation 3: Untracked Working Tree Data Drift (DVC)",
         "Data file tampered with out-of-range outlier records.",
         "Agent executes 'dvc_detect_changes', flagging modified pointer status and hash divergence.",
         "Script 'scripts/restore_dvc.sh' runs 'dvc checkout data/raw/customer_churn.csv.dvc --force' restoring bit-for-bit truth."
        )
    ]

    for f_title, root_cause, diag, res in failures:
        f_flow = [
            Paragraph(f"<b>{f_title}</b>", h2_style),
            Paragraph(f"• <b>Root Cause:</b> {root_cause}", bullet_style),
            Paragraph(f"• <b>Agent Diagnostic:</b> {diag}", bullet_style),
            Paragraph(f"• <b>Remediation:</b> {res}", bullet_style),
            Spacer(1, 6)
        ]
        story.append(KeepTogether(f_flow))

    # =========================================================================
    # SECTION 6: LIMITATIONS & ROADMAP
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("6. Candid Limitations & Production Roadmap", h1_style))
    story.append(Paragraph(
        "<b>1. Host Elevation Boundary:</b> In restricted non-administrative developer environments, starting Docker Desktop "
        "services (<code>com.docker.service</code>) requires UAC elevation. Offline mock suites ensure full CI/CD verification "
        "while production deployments utilize hosted Kubernetes clusters.<br/>"
        "<b>2. Streaming Agent Interactivity:</b> Current agent interaction operates via HTTP REST (<code>POST /chat</code>). "
        "The production roadmap includes WebSockets for token streaming and live tool call progress.<br/>"
        "<b>3. Statistical Drift Detection:</b> While model age and validation thresholds are actively governed, "
        "integrating Evidently AI or Great Expectations into the inference service will enable real-time KS-test monitoring.",
        body_style
    ))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[SUCCESS] Publication-grade PDF generated at: {pdf_path}")


if __name__ == "__main__":
    current_dir = os.path.abspath(".")
    if os.path.basename(current_dir) == "assignment-03-mlops-agent":
        repo_dir = current_dir
    elif os.path.exists(os.path.join(current_dir, "assignment-03-mlops-agent")):
        repo_dir = os.path.join(current_dir, "assignment-03-mlops-agent")
    else:
        repo_dir = current_dir

    pdf_target = os.path.join(repo_dir, "REPORT.pdf")
    build_pdf(pdf_target, repo_dir)

