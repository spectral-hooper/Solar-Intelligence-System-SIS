#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIS v8 — Solar Intelligence System GUI
========================================

A PySide6 desktop application wrapping the Hinode SP Sigma-V Analyzer
backend (`solar_analyzer_SIS_v8_fixed.analyze_fits_file`).

Architecture
------------
- Tab 1  "Control Center"   — session naming, file selection, launch, live console
- Tab 2  "Master Dashboard" — auto-loads the generated dashboard PNG
- Tab 3  "AI Report"        — auto-loads the LLM scientific report
- Tab 4  "Repository"       — QTreeView file explorer over SIS_Runs/

The backend runs on a QThread so the GUI never freezes.
stdout/stderr are redirected to the live console via Qt signals.
"""

import os
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import (
    Qt,
    QObject,
    QThread,
    QDir,
    Signal,
    Slot,
    QSize,
    QUrl,
)
from PySide6.QtGui import (
    QPixmap,
    QFont,
    QIcon,
    QColor,
    QPalette,
    QFontDatabase,
    QDesktopServices,
)
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QFrame,
    QGraphicsDropShadowEffect,
    QMessageBox,
    QTreeView,
    QHeaderView,
)

# QFileSystemModel lives in QtWidgets on PySide6 >= 6.5
try:
    from PySide6.QtWidgets import QFileSystemModel
except ImportError:
    from PySide6.QtCore import QFileSystemModel  # type: ignore[attr-defined]


# ============================================================================
# WORKSPACE ROOT — all session outputs are stored here
# ============================================================================

WORKSPACE_ROOT = os.path.join(os.getcwd(), "SIS_Runs")


# ============================================================================
# CUSTOM DARK STYLESHEET  (SpaceX / NASA mission-control aesthetic)
# ============================================================================

DARK_STYLESHEET = """
/* -- Global ------------------------------------------------------------ */
QMainWindow, QWidget {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* -- Tab widget -------------------------------------------------------- */
QTabWidget::pane {
    border: 1px solid #21262d;
    border-radius: 6px;
    background: #0d1117;
}
QTabBar::tab {
    background: #161b22;
    color: #8b949e;
    border: 1px solid #21262d;
    border-bottom: none;
    padding: 10px 28px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    min-width: 150px;
}
QTabBar::tab:selected {
    background: #0d1117;
    color: #58a6ff;
    border-bottom: 2px solid #58a6ff;
}
QTabBar::tab:hover:!selected {
    background: #1c2128;
    color: #c9d1d9;
}

/* -- Buttons ----------------------------------------------------------- */
QPushButton {
    background-color: #21262d;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #30363d;
    border-color: #58a6ff;
    color: #ffffff;
}
QPushButton:pressed {
    background-color: #0d419d;
}
QPushButton:disabled {
    background-color: #161b22;
    color: #484f58;
    border-color: #21262d;
}

QPushButton#launchBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb, stop:1 #58a6ff);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 14px 40px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
QPushButton#launchBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #388bfd, stop:1 #79c0ff);
}
QPushButton#launchBtn:pressed {
    background: #1158c7;
}
QPushButton#launchBtn:disabled {
    background: #21262d;
    color: #484f58;
}

/* -- Line edits -------------------------------------------------------- */
QLineEdit {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: #1f6feb;
}
QLineEdit:focus {
    border-color: #58a6ff;
}

/* -- Text edit (console) ----------------------------------------------- */
QTextEdit {
    background-color: #010409;
    color: #7ee787;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 8px;
    font-family: "JetBrains Mono", "Fira Code", "Cascadia Code",
                 "SF Mono", "Consolas", monospace;
    font-size: 12px;
    selection-background-color: #1f6feb;
}
QTextEdit#reportText {
    color: #c9d1d9;
    font-family: "Segoe UI", "Roboto", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    line-height: 1.5;
}

/* -- Progress bar ------------------------------------------------------ */
QProgressBar {
    background-color: #161b22;
    border: 1px solid #21262d;
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1f6feb, stop:0.5 #58a6ff, stop:1 #1f6feb);
    border-radius: 4px;
}

/* -- Scroll area ------------------------------------------------------- */
QScrollArea {
    background: #0d1117;
    border: none;
}
QScrollBar:vertical {
    background: #0d1117;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #30363d;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #484f58;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #0d1117;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #30363d;
    border-radius: 5px;
    min-width: 30px;
}
QScrollBar::handle:horizontal:hover {
    background: #484f58;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* -- Labels ------------------------------------------------------------ */
QLabel#headerLabel {
    color: #58a6ff;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 1.5px;
}
QLabel#subLabel {
    color: #8b949e;
    font-size: 12px;
}
QLabel#statusLabel {
    color: #f0883e;
    font-size: 12px;
    font-weight: 600;
}
QLabel#waitingLabel {
    color: #484f58;
    font-size: 16px;
    font-style: italic;
}

/* -- Frames / separators ----------------------------------------------- */
QFrame#separator {
    background-color: #21262d;
    max-height: 1px;
}

/* -- Tree view (Repository tab) ---------------------------------------- */
QTreeView {
    background-color: #0d1117;
    color: #c9d1d9;
    border: 1px solid #21262d;
    border-radius: 6px;
    font-size: 13px;
    alternate-background-color: #161b22;
    outline: none;
}
QTreeView::item {
    padding: 4px 2px;
    border: none;
}
QTreeView::item:selected {
    background-color: #1f6feb;
    color: #ffffff;
}
QTreeView::item:hover:!selected {
    background-color: #1c2128;
}
QTreeView::branch {
    background: #0d1117;
}
QHeaderView::section {
    background-color: #161b22;
    color: #8b949e;
    border: 1px solid #21262d;
    padding: 6px 10px;
    font-weight: 600;
    font-size: 12px;
}
"""


# ============================================================================
# STDOUT / STDERR REDIRECT  ->  Qt signal
# ============================================================================


class _StreamRedirect(QObject):
    """Captures writes to a stream and emits them as Qt signals."""

    text_written = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._buffer = ""

    def write(self, text: str) -> None:
        if text:
            self.text_written.emit(str(text))

    def flush(self) -> None:
        pass


# ============================================================================
# WORKER THREAD -- runs the backend analysis off the main thread
# ============================================================================


class AnalysisWorker(QObject):
    """
    Executes ``analyze_fits_file`` in a background thread.

    Signals
    -------
    finished : str
        Emitted with the output_prefix on success.
    error : str
        Emitted with the traceback string on failure.
    """

    finished = Signal(str)  # output_prefix
    error = Signal(str)     # traceback text

    def __init__(self, fits_path: str, output_prefix: str = "adapt_adapt"):
        super().__init__()
        self.fits_path = fits_path
        self.output_prefix = output_prefix

    @Slot()
    def run(self) -> None:
        try:
            # Import the backend *inside* the worker so the heavy module
            # load (astropy, scipy, sklearn) doesn't block the GUI startup.
            from NEW_solar_analyzer_SIS_v8_fixed import analyze_fits_file

            analyze_fits_file(
                self.fits_path,
                output_prefix=self.output_prefix,
                n_mc_iterations=100,
            )
            self.finished.emit(self.output_prefix)
        except Exception:
            self.error.emit(traceback.format_exc())


# ============================================================================
# TAB 1 -- CONTROL CENTER
# ============================================================================


class ControlCenterTab(QWidget):
    """Main interaction hub: session naming, file picker, launch, live console."""

    # Emits (fits_path, session_name) when the user clicks Launch
    launch_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # -- Logo / header -------------------------------------------------
        header_frame = QWidget()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.setSpacing(4)

        # Try to load logo.png; fallback to styled text
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = Path(__file__).parent / "logo.png"
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            logo_label.setPixmap(
                pix.scaledToHeight(80, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            logo_label.setText("☀  SOLAR INTELLIGENCE SYSTEM")
            logo_label.setObjectName("headerLabel")

        subtitle = QLabel("Hinode SP Spectropolarimetric Analysis Pipeline  —  SIS v8")
        subtitle.setObjectName("subLabel")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_layout.addWidget(logo_label)
        header_layout.addWidget(subtitle)
        layout.addWidget(header_frame)

        # -- Separator -----------------------------------------------------
        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        # -- Session / analysis name (NEW) ---------------------------------
        session_label = QLabel("SESSION / ANALYSIS NAME")
        session_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-weight: 700; "
            "letter-spacing: 1.5px; margin-top: 2px; margin-bottom: 0px;"
        )
        layout.addWidget(session_label)

        self.session_edit = QLineEdit()
        self.session_edit.setPlaceholderText(
            "e.g., Flare_Region_Active   (leave empty for auto-generated name)"
        )
        self.session_edit.setMinimumHeight(38)
        layout.addWidget(self.session_edit)

        # -- File selection row --------------------------------------------
        file_label_layout = QHBoxLayout()

        file_label = QLabel("FITS INPUT FILE")
        file_label.setStyleSheet(
            "color: #8b949e; font-size: 11px; font-weight: 700; "
            "letter-spacing: 1.5px; margin-top: 6px; margin-bottom: 0px;"
        )

        archive_link = QLabel(
            "<a href='https://data.darts.isas.jaxa.jp/pub/hinode/sot/level1hao/' "
            "style='color: #58a6ff; text-decoration: none;'>Download FITS from Hinode Archive ↗</a>"
        )
        archive_link.setOpenExternalLinks(True)
        archive_link.setStyleSheet("margin-top: 6px; margin-bottom: 0px; font-size: 11px; font-weight: 600;")
        archive_link.setCursor(Qt.CursorShape.PointingHandCursor)

        file_label_layout.addWidget(file_label)
        file_label_layout.addStretch()
        file_label_layout.addWidget(archive_link)

        layout.addLayout(file_label_layout)

        file_row = QHBoxLayout()
        file_row.setSpacing(8)

        self.fits_edit = QLineEdit()
        self.fits_edit.setPlaceholderText("Path to FITS file ...")
        self.fits_edit.setMinimumHeight(38)

        browse_btn = QPushButton("  Select FITS File")
        browse_btn.setMinimumHeight(38)
        browse_btn.clicked.connect(self._browse_fits)

        file_row.addWidget(self.fits_edit, stretch=1)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # -- Launch button -------------------------------------------------
        self.launch_btn = QPushButton("🚀   LAUNCH SIS ANALYSIS")
        self.launch_btn.setObjectName("launchBtn")
        self.launch_btn.setMinimumHeight(52)
        self.launch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.launch_btn.clicked.connect(self._on_launch)

        # Glow effect
        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(30)
        glow.setColor(QColor(88, 166, 255, 90))
        glow.setOffset(0, 0)
        self.launch_btn.setGraphicsEffect(glow)

        layout.addWidget(self.launch_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # -- Progress bar --------------------------------------------------
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        self.progress.setFixedHeight(8)
        layout.addWidget(self.progress)

        # -- Status label --------------------------------------------------
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # -- Live console --------------------------------------------------
        console_label = QLabel("LIVE CONSOLE")
        console_label.setStyleSheet(
            "color: #484f58; font-size: 11px; font-weight: 700; "
            "letter-spacing: 2px; margin-top: 4px;"
        )
        layout.addWidget(console_label)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(200)
        layout.addWidget(self.console, stretch=1)

    # -- Slots -------------------------------------------------------------

    def _browse_fits(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select FITS File",
            "",
            "FITS Files (*.fits *.fits.gz *.fts);;All Files (*)",
        )
        if path:
            self.fits_edit.setText(path)

    def _on_launch(self) -> None:
        fits_path = self.fits_edit.text().strip()
        if not fits_path:
            self.append_log("[ERROR] No FITS file selected.\n")
            return
        if not os.path.isfile(fits_path):
            self.append_log(f"[ERROR] File not found: {fits_path}\n")
            return

        # Read or auto-generate session name
        session_name = self.session_edit.text().strip()
        if not session_name:
            session_name = f"Session_{time.strftime('%Y%m%d_%H%M%S')}"
            self.session_edit.setText(session_name)

        # Sanitise: replace problematic characters with underscores
        for ch in (" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"):
            session_name = session_name.replace(ch, "_")

        self.launch_requested.emit(fits_path, session_name)

    def set_running(self, running: bool) -> None:
        """Toggle UI elements between running / idle state."""
        self.launch_btn.setEnabled(not running)
        self.progress.setVisible(running)
        if running:
            self.status_label.setText("⏳  Analysis in progress ...")
            self.status_label.setStyleSheet(
                "color: #d29922; font-size: 12px; font-weight: 600;"
            )
        else:
            self.status_label.setText("✅  Analysis complete")
            self.status_label.setStyleSheet(
                "color: #3fb950; font-size: 12px; font-weight: 600;"
            )

    def set_error_status(self, msg: str) -> None:
        self.status_label.setText(f"❌  {msg}")
        self.status_label.setStyleSheet(
            "color: #f85149; font-size: 12px; font-weight: 600;"
        )

    @Slot(str)
    def append_log(self, text: str) -> None:
        """Append text to the live console (called from stdout redirect)."""
        cursor = self.console.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()


# ============================================================================
# TAB 2 -- MASTER DASHBOARD
# ============================================================================


class DashboardTab(QWidget):
    """Displays the generated Master Dashboard image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_label = QLabel("Waiting for analysis completion ...")
        self.image_label.setObjectName("waitingLabel")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

        self._pixmap = None

    def load_image(self, path: str) -> None:
        """Load and display a high-res dashboard image."""
        if not os.path.isfile(path):
            self.image_label.setText(f"Dashboard image not found:\n{path}")
            return

        self._pixmap = QPixmap(path)
        if self._pixmap.isNull():
            self.image_label.setText(f"Failed to load image:\n{path}")
            return

        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        available = self.scroll_area.viewport().size()
        scaled = self._pixmap.scaled(
            available,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_scaled_pixmap()


# ============================================================================
# TAB 3 -- AI REPORT
# ============================================================================


class ReportTab(QWidget):
    """Displays the AI-generated scientific report."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("AI SCIENTIFIC REPORT")
        title.setStyleSheet(
            "color: #58a6ff; font-size: 15px; font-weight: 700; "
            "letter-spacing: 2px; margin-bottom: 8px;"
        )
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.report_text = QTextEdit()
        self.report_text.setObjectName("reportText")
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText(
            "The AI Scientific Report will appear here after analysis ..."
        )
        layout.addWidget(self.report_text)

    def load_report(self, path: str) -> None:
        if not os.path.isfile(path):
            self.report_text.setPlainText(
                "AI Report not generated.\n\n"
                "This typically means the Ollama LLM was not running during "
                "analysis.  The quantitative results in the CSV and dashboard "
                "are unaffected."
            )
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
            self.report_text.setPlainText(text)
        except Exception as exc:
            self.report_text.setPlainText(f"Error reading report:\n{exc}")


# ============================================================================
# TAB 4 -- REPOSITORY  (file explorer over SIS_Runs/)
# ============================================================================


class RepositoryTab(QWidget):
    """
    Internal file explorer rooted at the ``SIS_Runs/`` workspace folder.

    Uses ``QFileSystemModel`` + ``QTreeView`` to let the user browse all
    past and current analysis sessions, including ``examples/``,
    ``anomalies/``, CSVs, and generated PNGs.
    """

    def __init__(self, workspace_root: str, parent=None):
        super().__init__(parent)
        self._workspace_root = workspace_root
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # -- Header row ----------------------------------------------------
        header_row = QHBoxLayout()

        title = QLabel("ANALYSIS WORKSPACE")
        title.setStyleSheet(
            "color: #58a6ff; font-size: 15px; font-weight: 700; "
            "letter-spacing: 2px;"
        )

        path_label = QLabel(self._workspace_root)
        path_label.setStyleSheet(
            "color: #484f58; font-size: 11px; font-style: italic;"
        )

        refresh_btn = QPushButton("⟳  Refresh")
        refresh_btn.setFixedWidth(100)
        refresh_btn.clicked.connect(self.refresh)

        header_row.addWidget(title)
        header_row.addStretch()
        header_row.addWidget(path_label)
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        # -- Tree view -----------------------------------------------------
        self.fs_model = QFileSystemModel()
        self.fs_model.setReadOnly(True)

        # Ensure root directory exists before setting it
        os.makedirs(self._workspace_root, exist_ok=True)

        self.fs_model.setRootPath(self._workspace_root)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.fs_model)
        self.tree_view.setRootIndex(self.fs_model.index(self._workspace_root))
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.setAnimated(True)
        self.tree_view.setIndentation(20)
        self.tree_view.doubleClicked.connect(self._open_repository_file)

        # Column sizing
        header = self.tree_view.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, self.fs_model.columnCount()):
            header.setSectionResizeMode(
                col, QHeaderView.ResizeMode.ResizeToContents
            )

        layout.addWidget(self.tree_view, stretch=1)

    def refresh(self) -> None:
        """Re-scan the workspace (called after a new analysis completes)."""
        os.makedirs(self._workspace_root, exist_ok=True)
        self.fs_model.setRootPath(self._workspace_root)
        self.tree_view.setRootIndex(
            self.fs_model.index(self._workspace_root)
        )
    def _open_repository_file(self, index) -> None:
        file_path = self.fs_model.filePath(index)
        if os.path.isfile(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

# ============================================================================
# MAIN WINDOW
# ============================================================================


class SISMainWindow(QMainWindow):
    """Top-level window for the Solar Intelligence System GUI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SIS v8  ---  Solar Intelligence System")
        self.setMinimumSize(1100, 750)
        self.resize(1280, 850)

        # Track the current session name for completion notifications
        self._current_session_name: str = ""

        # -- Central widget with tabs --------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        self.tab_control = ControlCenterTab()
        self.tab_dashboard = DashboardTab()
        self.tab_report = ReportTab()
        self.tab_repository = RepositoryTab(WORKSPACE_ROOT)

        self.tabs.addTab(self.tab_control, "⚡  Control Center")
        self.tabs.addTab(self.tab_dashboard, "📊  Master Dashboard")
        self.tabs.addTab(self.tab_report, "📝  AI Report")
        self.tabs.addTab(self.tab_repository, "🗄️  Repository")

        main_layout.addWidget(self.tabs)

        # -- Footer --------------------------------------------------------
        footer = QLabel(
            "Solar Intelligence System  |  Hinode/SP Pipeline  |  "
            "Powered by Gemini 1.5 Flash + Physics Engines"
        )
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(
            "color: #30363d; font-size: 10px; padding: 4px;"
        )
        main_layout.addWidget(footer)

        # -- Stdout / stderr redirect --------------------------------------
        self._stdout_redirect = _StreamRedirect()
        self._stderr_redirect = _StreamRedirect()
        self._stdout_redirect.text_written.connect(self.tab_control.append_log)
        self._stderr_redirect.text_written.connect(self.tab_control.append_log)
        sys.stdout = self._stdout_redirect
        sys.stderr = self._stderr_redirect

        # -- Worker thread (lazy) ------------------------------------------
        self._worker_thread: QThread | None = None
        self._worker: AnalysisWorker | None = None

        # -- Connections ---------------------------------------------------
        self.tab_control.launch_requested.connect(self._start_analysis)

    # -- Analysis lifecycle ------------------------------------------------

    @Slot(str, str)
    def _start_analysis(self, fits_path: str, session_name: str) -> None:
        """Create the session directory and spin up a worker thread."""
        try:
            if self._worker_thread is not None and self._worker_thread.isRunning():
                self.tab_control.append_log(
                    "[WARN] Analysis already running. Please wait.\n"
                )
                return
        except RuntimeError:
            self._worker_thread = None

        # -- Build the organised output path -------------------------------
        # Structure:  SIS_Runs/<session_name>/<session_name>_*
        self._current_session_name = session_name
        session_dir = os.path.join(WORKSPACE_ROOT, session_name)
        os.makedirs(session_dir, exist_ok=True)
        output_prefix = os.path.join(session_dir, session_name)

        self.tab_control.set_running(True)
        self.tab_control.console.clear()
        self.tab_control.append_log(
            f"{'='*70}\n"
            f"  SIS v8 --- Solar Intelligence System\n"
            f"  Session : {session_name}\n"
            f"  FITS    : {fits_path}\n"
            f"  Output  : {session_dir}/\n"
            f"{'='*70}\n\n"
        )

        # Create worker + thread
        self._worker_thread = QThread()
        self._worker = AnalysisWorker(fits_path, output_prefix)
        self._worker.moveToThread(self._worker_thread)

        # Wire signals
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)

        # Clean up after thread exits
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(self._worker.deleteLater)

        self._worker_thread.start()

    @Slot(str)
    def _on_analysis_finished(self, prefix: str) -> None:
        """Load outputs, switch to Dashboard, refresh Repository, notify."""
        self.tab_control.set_running(False)
        self.tab_control.append_log(
            "\n[GUI] Analysis complete.  Loading outputs ...\n"
        )

        # Load dashboard image
        dash_path = f"{prefix}_SIS_v3_Master_Dashboard.png"
        self.tab_dashboard.load_image(dash_path)
        self.tab_control.append_log(f"[GUI] Dashboard loaded: {dash_path}\n")

        # Load AI report
        report_path = f"{prefix}_AI_SCIENTIFIC_REPORT.txt"
        self.tab_report.load_report(report_path)
        self.tab_control.append_log(f"[GUI] AI Report loaded: {report_path}\n")

        # Refresh the Repository tree to show the new session folder
        self.tab_repository.refresh()

        # Switch to Dashboard tab (the "wow" moment) --- PRESERVED
        self.tabs.setCurrentIndex(1)

        # Completion notification dialog
        session_dir = os.path.join(WORKSPACE_ROOT, self._current_session_name)
        QMessageBox.information(
            self,
            "Analysis Complete",
            f"Analysis Complete!\n\n"
            f"The new analysis repository "
            f"'{self._current_session_name}' "
            f"is ready and saved in the workspace.\n\n"
            f"Location:\n{session_dir}",
        )

    @Slot(str)
    def _on_analysis_error(self, tb_text: str) -> None:
        """Display error in the console and stop the progress bar."""
        self.tab_control.set_running(False)
        self.tab_control.set_error_status("Analysis failed --- see console")
        self.tab_control.append_log(
            f"\n{'='*60}\n"
            f"[ERROR] Analysis crashed:\n"
            f"{'='*60}\n"
            f"{tb_text}\n"
        )

    # -- Cleanup -----------------------------------------------------------

def closeEvent(self, event) -> None:
        """Restore stdout/stderr before exiting."""
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        try:
            if self._worker_thread and self._worker_thread.isRunning():
                self._worker_thread.quit()
                self._worker_thread.wait(3000)
        except RuntimeError:
            pass
        super().closeEvent(event)


# ============================================================================
# ENTRY POINT
# ============================================================================


def main() -> None:
    # Ensure the backend module can be found (same directory)
    backend_dir = str(Path(__file__).resolve().parent)
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # cross-platform consistent look

    # Apply the custom dark stylesheet
    app.setStyleSheet(DARK_STYLESHEET)

    window = SISMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
