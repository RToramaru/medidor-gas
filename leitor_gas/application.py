import ctypes
import hashlib
import json
import queue
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
import tkinter.filedialog as filedialog
from tkinter import messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from leitor_gas.settings import (
    APP_ICON_PATH,
    BACKGROUND,
    BORDER,
    CALIBRATIONS_DIR,
    DECIMAL_CLASSIFIER,
    GREEN,
    INTEGER_CLASSIFIER,
    MUTED,
    PANEL,
    RED,
    RESULTS_DIR,
    TEXT,
)
from leitor_gas.persistence import (
    load_calibration,
    migrate_legacy_results,
    save_calibration,
    update_calibration_regions,
)
from leitor_gas.calibration_preview import (
    render_calibration_preview,
    suggest_region_coordinates,
)
from leitor_gas.cameras import find_cameras
from leitor_gas.vision import classify_roi, detect_digit_positions, load_classifier


class MeterCard(ctk.CTkFrame):
    def __init__(
        self,
        master,
        app,
        card_id: int,
        source: str | int,
        source_type: str,
        title: str,
    ):
        super().__init__(
            master,
            fg_color=PANEL,
            corner_radius=14,
            border_width=1,
            border_color=BORDER,
        )
        self.app = app
        self.card_id = card_id
        self.source = source
        self.source_type = source_type
        self.title_text = title
        self.capture = None
        self.running = False
        self.paused = False
        self.closed = False
        self.frame = None
        self.calibration = None
        self.last_values = None
        self.last_reading = "—"
        self.initialized_reading = False
        self.calibrating = False
        self.calibration_attempted = False
        self.calibration_request_pending = False
        self.resolution_mismatch = False
        self.reading_error = False
        self.writer = None
        self.recording = False
        self.image_ref = None
        self.worker_stop = threading.Event()
        self.recording_lock = threading.Lock()
        self.analysis_lock = threading.Lock()
        self.capture_thread = None
        self.analysis_thread = None
        self.connecting = False
        self.source_finished = False
        self.frame_queue = Queue(maxsize=1)
        self.source_frame_interval = 0.0

        source_key = f"{source_type}:{source}"
        source_hash = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
        self.calibration_file = CALIBRATIONS_DIR / f"calibracao_{source_hash}.json"

        self._build_card()
        self._load_calibration()
        self._open_source()

    def _build_card(self):
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        header.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            header,
            text=self.title_text,
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        )
        self.title_label.grid(row=0, column=0, sticky="ew")
        self.status_label = ctk.CTkLabel(
            header, text="●  Aguardando", text_color=MUTED
        )
        self.status_label.grid(row=0, column=1, padx=8)
        ctk.CTkButton(
            header,
            text="×",
            width=30,
            height=28,
            fg_color="#353A44",
            hover_color=RED,
            command=self.close,
        ).grid(row=0, column=2)

        self.preview = ctk.CTkLabel(
            self,
            text="A imagem aparecerá aqui",
            fg_color="#080A0E",
            corner_radius=9,
            height=235,
            text_color=MUTED,
        )
        self.preview.grid(row=1, column=0, sticky="nsew", padx=14, pady=5)

        reading_box = ctk.CTkFrame(self, fg_color="#10251B", corner_radius=10)
        reading_box.grid(row=2, column=0, sticky="ew", padx=14, pady=7)
        reading_box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            reading_box,
            text="LEITURA ATUAL",
            text_color="#8EB99C",
            font=ctk.CTkFont(size=10, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))
        self.reading_label = ctk.CTkLabel(
            reading_box,
            text="—",
            text_color="#55E58A",
            font=ctk.CTkFont(size=30, weight="bold"),
        )
        self.reading_label.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))

        self.info_label = ctk.CTkLabel(
            self,
            text="A leitura começa automaticamente.",
            text_color=MUTED,
            anchor="w",
            justify="left",
            wraplength=460,
        )
        self.info_label.grid(row=3, column=0, sticky="ew", padx=14, pady=(2, 8))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 13))
        for column in range(2):
            actions.grid_columnconfigure(column, weight=1)

        self.pause_button = None
        self.record_button = None
        self.calibrate_button = ctk.CTkButton(
            actions,
            text="Calibrar novamente",
            height=34,
            command=self.calibrate,
        )
        if self.source_type == "video":
            self.pause_button = ctk.CTkButton(
                actions,
                text="Ⅱ  Pausar",
                height=34,
                fg_color="#414752",
                hover_color="#555D6B",
                command=self.toggle_pause,
            )
            self.pause_button.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
            self.calibrate_button.grid(
                row=0, column=1, padx=3, pady=3, sticky="ew"
            )
        elif self.source_type == "camera":
            self.record_button = ctk.CTkButton(
                actions,
                text="Gravar vídeo",
                height=34,
                fg_color="#B64B35",
                hover_color="#D45B42",
                command=self.toggle_recording,
            )
            self.record_button.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
            self.calibrate_button.grid(
                row=0, column=1, padx=3, pady=3, sticky="ew"
            )
        else:
            self.calibrate_button.grid(
                row=0, column=0, columnspan=2, padx=3, pady=3, sticky="ew"
            )

        self.reconnect_button = ctk.CTkButton(
            actions,
            text="Reconectar câmera",
            height=34,
            command=self.reconnect_camera,
            state="disabled",
        )
        self.reconnect_button.grid(
            row=2, column=0, columnspan=2, padx=3, pady=3, sticky="ew"
        )
        self.reconnect_button.grid_remove()
        self.edit_calibration_button = ctk.CTkButton(
            actions,
            text="Ajustar regiões",
            height=34,
            command=self.open_calibration_editor,
            state="disabled",
        )
        self.edit_calibration_button.grid(
            row=1, column=0, columnspan=2, padx=3, pady=3, sticky="ew"
        )

    def _open_source(self):
        if self.source_type == "image":
            try:
                image_data = np.fromfile(str(self.source), dtype=np.uint8)
            except OSError as exc:
                self._set_status("Não foi possível abrir a imagem.", error=True)
                messagebox.showerror(
                    "Erro ao abrir imagem", str(exc), parent=self.app
                )
                return
            self.frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if self.frame is None:
                self._set_status("Não foi possível abrir a imagem.", error=True)
                return

            self._show_frame(self.frame)
            if self.calibration is not None:
                self._start_image_analysis(self.frame)
            else:
                self.after(250, self.calibrate)
            return

        if self.source_type == "camera":
            self.reconnect_button.grid()
        self._connect_source()

    def _connect_source(self):
        if (
            self.closed
            or self.connecting
            or self.capture_thread is not None and self.capture_thread.is_alive()
            or self.analysis_thread is not None and self.analysis_thread.is_alive()
        ):
            return
        self.connecting = True
        self.worker_stop.clear()
        self._set_status("Conectando à fonte…")
        threading.Thread(target=self._connect_source_worker, daemon=True).start()

    def _connect_source_worker(self):
        video_source = self.source if self.source_type == "camera" else str(self.source)
        try:
            capture = cv2.VideoCapture(video_source)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError("Não foi possível abrir esta fonte.")
        except Exception as exc:
            self.app.schedule_ui(lambda error=exc: self._source_connection_failed(error))
            return
        self.app.schedule_ui(lambda: self._source_connected(capture))

    def _source_connected(self, capture):
        self.connecting = False
        if self.closed:
            capture.release()
            return
        self.capture = capture
        self.running = True
        self.source_finished = False
        self.frame_queue = Queue(maxsize=1)
        if self.source_type == "video":
            fps = capture.get(cv2.CAP_PROP_FPS)
            self.source_frame_interval = 1.0 / fps if 1 <= fps <= 120 else 1 / 30
        else:
            self.source_frame_interval = 0.0
        if self.source_type == "camera":
            with self.analysis_lock:
                self.last_values = None
                self.initialized_reading = False
                self.last_reading = "—"
            self.reading_label.configure(text="—")
        self.reconnect_button.grid_remove()
        self._set_status("Ao vivo", color=GREEN)
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()
        self.analysis_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.analysis_thread.start()

    def _source_connection_failed(self, error: Exception):
        self.connecting = False
        if self.closed:
            return
        self.running = False
        self._set_status(str(error), error=True)
        if self.source_type == "camera":
            self.reconnect_button.configure(state="normal")
            self.reconnect_button.grid()

    def reconnect_camera(self):
        if self.source_type != "camera" or self.running:
            return
        self.reconnect_button.configure(state="disabled")
        self._connect_source()

    def _set_status(self, text: str, error: bool = False, color: str | None = None):
        if self.closed:
            return
        if color is None:
            color = RED if error else MUTED
        self.status_label.configure(
            text=f"●  {'Atenção' if error else text}",
            text_color=color,
        )
        self.info_label.configure(text=text)

    def _load_calibration(self):
        try:
            data = load_calibration(self.calibration_file)
            if data is None:
                return
            self.calibration = data
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._set_status(f"Não foi possível carregar a calibração: {exc}", error=True)

    def _stream_worker(self):
        while not self.worker_stop.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                if self.source_finished or not self.running:
                    break
                continue

            if self.worker_stop.is_set():
                break
            if self.calibration is None:
                self._schedule_preview(frame)
                if (
                    not self.calibrating
                    and not self.calibration_attempted
                    and not self.calibration_request_pending
                ):
                    self.calibration_request_pending = True
                    self.app.schedule_ui(
                        lambda image=frame.copy(): self._request_calibration(image)
                    )
            elif self._compatible(frame):
                self.resolution_mismatch = False
                if self.reading_error:
                    self._schedule_preview(frame)
                else:
                    self._process_frame(frame, initialize=not self.initialized_reading)
            else:
                self._schedule_preview(frame)
                if not self.resolution_mismatch:
                    self.resolution_mismatch = True
                    self.app.schedule_latest_status(self, self._resolution_changed)

        if self.source_finished and not self.worker_stop.is_set():
            if self.source_type == "video":
                self.app.schedule_ui(self._video_finished)
            else:
                self.app.schedule_ui(self._camera_disconnected)

    def _capture_worker(self):
        while self.running and not self.worker_stop.is_set():
            started_at = time.monotonic()
            if self.paused:
                time.sleep(0.05)
                continue
            capture = self.capture
            if capture is None:
                break

            ok, frame = capture.read()
            if not ok:
                self.running = False
                self.source_finished = True
                capture.release()
                self.capture = None
                break

            self.frame = frame
            if self.recording:
                self._write_frame(frame)

            try:
                self.frame_queue.put_nowait(frame)
            except Full:
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
                self.frame_queue.put_nowait(frame)

            if self.source_frame_interval:
                elapsed = time.monotonic() - started_at
                time.sleep(max(0.0, self.source_frame_interval - elapsed))

    def _video_finished(self):
        self._release_writer()
        self._set_status("Vídeo finalizado.")

    def _camera_disconnected(self):
        self._release_writer()
        self._set_status("Câmera desconectada. Tente reconectar.", error=True)
        self.reconnect_button.configure(state="normal")
        self.reconnect_button.grid()

    def _write_frame(self, frame: np.ndarray):
        error = None
        with self.recording_lock:
            if self.writer is None:
                return
            try:
                self.writer.write(frame)
            except cv2.error as exc:
                error = exc
        if error is not None:
            self.app.schedule_ui(lambda exc=error: self._recording_failed(exc))

    def _recording_failed(self, error: Exception):
        self._release_writer()
        self._set_status("A gravação foi interrompida.", error=True)
        messagebox.showerror("Erro ao gravar", str(error), parent=self.app)

    def _resolution_changed(self):
        self._set_status("A resolução mudou. Ajuste as regiões novamente.", error=True)

    def _schedule_preview(self, frame: np.ndarray):
        self.app.schedule_latest_frame(self, frame)

    def _request_calibration(self, frame: np.ndarray):
        self.calibration_request_pending = False
        if self.closed or self.calibration is not None or self.calibrating:
            return
        self.frame = frame
        self.calibrate()

    def _start_image_analysis(self, frame: np.ndarray):
        self.reading_error = False
        self.last_values = None
        self.initialized_reading = False
        threading.Thread(
            target=self._image_analysis_worker, args=(frame,), daemon=True
        ).start()

    def _image_analysis_worker(self, frame: np.ndarray):
        if not self.closed:
            self._process_frame(frame, initialize=True)

    def _compatible(self, frame: np.ndarray) -> bool:
        if self.calibration is None:
            return False
        height, width = frame.shape[:2]
        return (
            int(self.calibration["width"]) == width
            and int(self.calibration["height"]) == height
        )

    def calibrate(self):
        if self.frame is None:
            return
        if self.calibrating:
            return

        self.calibrating = True
        self.calibration_attempted = True
        self.calibrate_button.configure(state="disabled")
        self._set_status("Localizando os dígitos…")
        frame = self.frame.copy()
        threading.Thread(
            target=self._calibrate_worker, args=(frame,), daemon=True
        ).start()

    def _calibrate_worker(self, frame: np.ndarray):
        try:
            boxes, backend = detect_digit_positions(frame)
            height, width = frame.shape[:2]
            rois = [
                {
                    "x1": int(box["x1"]),
                    "y1": int(box["y1"]),
                    "x2": int(box["x2"]),
                    "y2": int(box["y2"]),
                    "confidence": box["confidence"],
                    "name": box.get("name", ""),
                }
                for box in boxes
            ]
            data = {
                "width": width,
                "height": height,
                "backend": backend,
                "rois": rois,
            }
            if self.closed:
                return
            save_calibration(self.calibration_file, data)
        except Exception as exc:
            self.app.schedule_ui(lambda error=exc: self._calibration_failed(error))
            return

        self.app.schedule_ui(lambda: self._calibration_finished(data))

    def _calibration_finished(self, data: dict):
        self.calibrating = False
        if self.closed:
            return

        self.calibrate_button.configure(state="normal")
        with self.analysis_lock:
            self.calibration = data
            self.last_values = None
            self.initialized_reading = False
            self.reading_error = False
        self.edit_calibration_button.configure(state="normal")
        self.last_reading = "—"
        self.reading_label.configure(text="—")
        self.edit_calibration_button.configure(state="normal")
        if self.source_type == "image" and self.frame is not None:
            self._start_image_analysis(self.frame)
        else:
            self._set_status("Regiões localizadas. Iniciando leitura…")

    def _calibration_failed(self, error: Exception):
        self.calibrating = False
        if self.closed:
            return

        self.calibrate_button.configure(state="normal")
        self._set_status("Não foi possível calibrar esta fonte.", error=True)
        messagebox.showerror("Erro na calibração", str(error), parent=self.app)

    def _process_frame(self, frame: np.ndarray, initialize: bool = False):
        try:
            with self.analysis_lock, self.app.inference_lock:
                self._analyze_frame(frame, initialize)
        except Exception as exc:
            self.reading_error = True
            self.app.schedule_ui(lambda error=exc: self._analysis_failed(error))

    def _analyze_frame(self, frame: np.ndarray, initialize: bool = False):
        if self.closed or not self.calibration or not self.app.model11 or not self.app.model100:
            return
        if not self._compatible(frame):
            return

        height, width = frame.shape[:2]
        rois = self.calibration["rois"]
        if self.last_values is None or len(self.last_values) != len(rois):
            self.last_values = [None] * len(rois)

        last_roi = rois[-1]
        x1, y1, x2, y2 = self._clip_roi(last_roi, width, height)
        if x2 <= x1 or y2 <= y1:
            self.app.schedule_latest_analysis(
                self, frame, "—", "Aguardando uma leitura completa."
            )
            return

        last_class, _ = classify_roi(
            frame[y1:y2, x1:x2], self.app.model100
        )
        last_value = round(last_class / 10.0, 1)
        self.last_values[-1] = {
            "kind": "decimal",
            "class": last_class,
            "value": last_value,
        }

        update_integer_digits = initialize or last_value < 1.0
        if update_integer_digits:
            for index, roi in enumerate(rois[:-1]):
                x1, y1, x2, y2 = self._clip_roi(roi, width, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                class_index, _ = classify_roi(
                    frame[y1:y2, x1:x2], self.app.model11
                )
                value = None if class_index == 10 else class_index
                self.last_values[index] = {
                    "kind": "integer",
                    "class": class_index,
                    "value": value,
                }

        reading_is_complete = all(
            value is not None and value["value"] is not None
            for value in self.last_values
        )
        reading = "—"
        if reading_is_complete:
            integer_part = "".join(
                str(value["value"]) for value in self.last_values[:-1]
            )
            reading = integer_part + f"{self.last_values[-1]['value']:.1f}"

        self.last_reading = reading
        if initialize:
            status = "Leitura inicial concluída."
        elif update_integer_digits:
            status = "Leitura atualizada."
        else:
            status = "Acompanhando o último dígito."
        if initialize:
            self.initialized_reading = True
        self.app.schedule_latest_analysis(
            self,
            frame,
            self.last_reading,
            status,
        )

    def _analysis_failed(self, error: Exception):
        if self.closed:
            return
        self._release_writer()
        self._set_status("Não foi possível processar esta imagem.", error=True)
        messagebox.showerror("Erro na leitura", str(error), parent=self.app)

    def open_calibration_editor(self):
        if not self.calibration:
            return

        editor = ctk.CTkToplevel(self.app)
        editor.title("Ajustar regiões dos dígitos")
        editor.geometry("1180x740")
        editor.minsize(1000, 640)
        editor.transient(self.app)
        editor.grab_set()
        editor.grid_columnconfigure(0, weight=3)
        editor.grid_columnconfigure(1, weight=2)
        editor.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            editor,
            text=(
                f"Edite os limites numéricos de cada região "
                f"({self.calibration['width']} × {self.calibration['height']}). "
                "A região mais à direita é a parte decimal."
            ),
            anchor="w",
            justify="left",
        ).grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="ew")

        preview_label = ctk.CTkLabel(
            editor,
            text="Prévia indisponível",
            fg_color="#080A0E",
            corner_radius=9,
            text_color=MUTED,
        )
        preview_label.grid(
            row=1, column=0, padx=(16, 8), pady=8, sticky="nsew"
        )
        table = ctk.CTkScrollableFrame(editor)
        table.grid(row=1, column=1, padx=(8, 16), pady=8, sticky="nsew")
        preview_frame = self.frame.copy() if self.frame is not None else None
        fields = []
        region_metadata = []

        def update_preview(_event=None):
            self._update_calibration_preview(preview_label, preview_frame, fields)

        def rebuild_region_rows(regions):
            fields.clear()
            region_metadata.clear()
            for child in table.winfo_children():
                child.destroy()

            columns = ("Região", "X inicial", "Y inicial", "X final", "Y final")
            for column, title in enumerate(columns):
                ctk.CTkLabel(
                    table, text=title, font=ctk.CTkFont(weight="bold")
                ).grid(row=0, column=column, padx=4, pady=5)

            for row, roi in enumerate(regions, start=1):
                region_metadata.append(
                    {
                        key: value
                        for key, value in roi.items()
                        if key not in {"x1", "y1", "x2", "y2"}
                    }
                )
                ctk.CTkLabel(table, text=str(row)).grid(
                    row=row, column=0, padx=4, pady=3
                )
                row_fields = []
                for column, coordinate in enumerate(
                    ("x1", "y1", "x2", "y2"), start=1
                ):
                    entry = ctk.CTkEntry(table, width=72)
                    entry.insert(0, str(int(roi[coordinate])))
                    entry.grid(row=row, column=column, padx=3, pady=3)
                    entry.bind("<KeyRelease>", update_preview)
                    entry.bind("<<Paste>>", update_preview)
                    row_fields.append(entry)
                fields.append(row_fields)

                ctk.CTkButton(
                    table,
                    text="Remover",
                    width=62,
                    height=28,
                    fg_color="#414752",
                    hover_color=RED,
                    command=lambda index=row - 1: remove_region(index),
                ).grid(row=row, column=5, padx=3, pady=3)

            update_preview()

        def current_regions():
            return [
                {
                    "x1": int(row_fields[0].get()),
                    "y1": int(row_fields[1].get()),
                    "x2": int(row_fields[2].get()),
                    "y2": int(row_fields[3].get()),
                    **region_metadata[index],
                }
                for index, row_fields in enumerate(fields)
            ]

        def add_region():
            height = int(self.calibration["height"])
            width = int(self.calibration["width"])
            try:
                regions = current_regions()
                update_calibration_regions(
                    self.calibration,
                    [
                        (region["x1"], region["y1"], region["x2"], region["y2"])
                        for region in regions
                    ],
                )
                coordinates = suggest_region_coordinates(
                    [
                        (region["x1"], region["y1"], region["x2"], region["y2"])
                        for region in regions
                    ],
                    width,
                    height,
                )
            except ValueError as exc:
                messagebox.showerror("Regiões inválidas", str(exc), parent=editor)
                return

            regions.append(
                {
                    **dict(zip(("x1", "y1", "x2", "y2"), coordinates)),
                    "confidence": 0.0,
                    "name": "manual",
                }
            )
            regions.sort(key=lambda region: (region["x1"] + region["x2"]) / 2)
            rebuild_region_rows(regions)

        def remove_region(index):
            if len(fields) <= 1:
                messagebox.showwarning(
                    "Regiões dos dígitos",
                    "Mantenha pelo menos uma região.",
                    parent=editor,
                )
                return

            try:
                regions = current_regions()
            except ValueError as exc:
                messagebox.showerror("Regiões inválidas", str(exc), parent=editor)
                return
            del regions[index]
            rebuild_region_rows(regions)

        rebuild_region_rows(self.calibration["rois"])
        self._update_calibration_preview(preview_label, preview_frame, fields)

        buttons = ctk.CTkFrame(editor, fg_color="transparent")
        buttons.grid(row=2, column=0, columnspan=2, padx=12, pady=(4, 12), sticky="e")
        ctk.CTkButton(
            buttons,
            text="Adicionar ponto",
            command=add_region,
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            buttons,
            text="Cancelar",
            fg_color="#414752",
            command=editor.destroy,
        ).pack(side="right", padx=4)
        ctk.CTkButton(
            buttons,
            text="Salvar regiões",
            command=lambda: self._save_calibration_edits(
                fields, editor, region_metadata
            ),
        ).pack(side="right", padx=4)

    @staticmethod
    def _update_calibration_preview(preview_label, frame, fields):
        if frame is None:
            return

        regions = []
        for row_fields in fields:
            try:
                regions.append(tuple(int(field.get()) for field in row_fields))
            except ValueError:
                regions.append(None)

        image = render_calibration_preview(frame, regions)
        image_ref = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=image.size,
        )
        preview_label.image_ref = image_ref
        preview_label.configure(image=image_ref, text="")

    def _save_calibration_edits(self, fields: list, editor, region_metadata: list):
        try:
            coordinates = [
                tuple(int(field.get()) for field in fields_for_roi)
                for fields_for_roi in fields
            ]
            data = update_calibration_regions(
                self.calibration, coordinates, region_metadata
            )
            save_calibration(self.calibration_file, data)
        except (ValueError, OSError) as exc:
            messagebox.showerror("Regiões inválidas", str(exc), parent=editor)
            return

        with self.analysis_lock:
            self.calibration = data
            self.last_values = None
            self.reading_label.configure(text="—")
            self.last_reading = "—"
        self._set_status("Regiões salvas. Lendo novamente…")
        editor.destroy()
        if self.source_type == "image" and self.frame is not None:
            self._start_image_analysis(self.frame)

    def _render_analysis(
        self,
        frame: np.ndarray,
        reading: str,
        status: str,
    ):
        if self.closed:
            return
        self.last_reading = reading
        self.reading_label.configure(text=self.last_reading)
        self._show_frame(frame)
        self._set_status(status, color=GREEN)

    @staticmethod
    def _clip_roi(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
        x1 = max(0, min(width - 1, int(box["x1"])))
        y1 = max(0, min(height - 1, int(box["y1"])))
        x2 = max(0, min(width, int(box["x2"])))
        y2 = max(0, min(height, int(box["y2"])))
        return x1, y1, x2, y2

    def _show_frame(self, frame: np.ndarray):
        if self.closed or frame is None:
            return

        height, width = frame.shape[:2]
        scale = min(540 / width, 300 / height)
        resized = cv2.resize(
            frame,
            (max(1, int(width * scale)), max(1, int(height * scale))),
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.image_ref = ctk.CTkImage(
            light_image=image,
            dark_image=image,
            size=(image.width, image.height),
        )
        self.preview.configure(image=self.image_ref, text="")

    def toggle_pause(self):
        if self.source_type != "video" or self.pause_button is None:
            return

        self.paused = not self.paused
        self.pause_button.configure(text="▶  Retomar" if self.paused else "Ⅱ  Pausar")
        self.status_label.configure(
            text="●  Pausado" if self.paused else "●  Ao vivo",
            text_color=MUTED if self.paused else GREEN,
        )

    def toggle_recording(self):
        if self.source_type != "camera" or self.record_button is None:
            return
        if self.recording:
            self._release_writer()
            self.info_label.configure(text="Gravação salva.")
            return
        if self.frame is None:
            return

        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = RESULTS_DIR / (
                f"video_{self.card_id}_{datetime.now():%Y%m%d_%H%M%S_%f}.mp4"
            )
            height, width = self.frame.shape[:2]
            fps = 25.0
            if self.capture is not None:
                source_fps = self.capture.get(cv2.CAP_PROP_FPS)
                if 1 <= source_fps <= 120:
                    fps = source_fps

            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
        except (OSError, cv2.error) as exc:
            messagebox.showerror("Erro ao gravar", str(exc), parent=self.app)
            return
        if not writer.isOpened():
            writer.release()
            messagebox.showerror(
                "Erro ao gravar",
                "Não foi possível iniciar a gravação neste local.",
                parent=self.app,
            )
            return

        with self.recording_lock:
            self.writer = writer
            self.recording = True
        self.record_button.configure(text="Parar gravação", fg_color=RED)
        self.info_label.configure(text=f"Gravando: {output_path.name}")

    def _release_writer(self):
        with self.recording_lock:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
            self.recording = False
        if not self.closed and self.record_button is not None:
            self.record_button.configure(text="Gravar vídeo", fg_color="#B64B35")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.running = False
        self.worker_stop.set()
        self._release_writer()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.app.remove_card(self)


class HydrometerApp(ctk.CTk):
    def __init__(self):
        if sys.platform == "win32":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "LeitorGas.Desktop"
            )
        super().__init__()
        self.title("Leitor de hidrômetros")
        if sys.platform == "win32":
            self.iconbitmap(str(APP_ICON_PATH))
        with Image.open(APP_ICON_PATH) as icon:
            icon.load()
            icon.thumbnail((64, 64))
            self._icon_image = ImageTk.PhotoImage(icon.copy())
        self.iconphoto(True, self._icon_image)
        self.geometry("1380x900")
        self.minsize(1000, 680)
        self.configure(fg_color=BACKGROUND)

        self.cards = []
        self.next_id = 1
        self.model11 = None
        self.model100 = None
        self.closing = False
        self.source_buttons = []
        self.worker_callbacks = queue.Queue()
        self.inference_lock = threading.Lock()
        self.pending_ui_lock = threading.Lock()
        self.pending_ui_callbacks = {}
        self.queued_ui_keys = set()
        self.migration_error = migrate_legacy_results()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(100, self._load_models)
        self.after(50, self._poll_worker_callbacks)

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        sidebar = ctk.CTkFrame(
            self, width=228, corner_radius=0, fg_color="#171A20"
        )
        sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="◉  HIDRÔMETRO",
            font=ctk.CTkFont(size=21, weight="bold"),
            text_color=TEXT,
        ).pack(anchor="w", padx=18, pady=(26, 4))
        ctk.CTkLabel(
            sidebar,
            text="LEITURA E MONITORAMENTO",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=20, pady=(0, 26))

        ctk.CTkLabel(
            sidebar,
            text="ADICIONAR FONTE",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=18, pady=(0, 8))

        for text, command, color in (
            ("Abrir imagem", self.add_image, "#303641"),
            ("Abrir vídeo", self.add_video, "#303641"),
            ("Usar câmera", self.add_camera, "#1976D2"),
        ):
            button = ctk.CTkButton(
                sidebar,
                text=text,
                anchor="w",
                height=38,
                fg_color=color,
                command=command,
                state="disabled",
            )
            button.pack(fill="x", padx=12, pady=4)
            self.source_buttons.append(button)

        ctk.CTkFrame(sidebar, height=1, fg_color=BORDER).pack(
            fill="x", padx=14, pady=18
        )
        ctk.CTkLabel(
            sidebar,
            text="STATUS",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=MUTED,
        ).pack(anchor="w", padx=18, pady=(0, 8))
        self.app_status = ctk.CTkLabel(
            sidebar,
            text="Preparando o leitor…",
            justify="left",
            anchor="w",
            wraplength=185,
            text_color=MUTED,
        )
        self.app_status.pack(fill="x", padx=18)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=1, sticky="ew", padx=22, pady=(20, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Painel de leitura",
            font=ctk.CTkFont(size=25, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            header,
            text="Acompanhe imagens, vídeos e câmera em tempo real",
            text_color=MUTED,
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.counter_label = ctk.CTkLabel(
            header,
            text="0 fontes",
            text_color="#9BC6FF",
            fg_color="#1D2A3B",
            corner_radius=8,
        )
        self.counter_label.grid(row=0, column=1, rowspan=2, padx=(12, 0))

        self.scroll = ctk.CTkScrollableFrame(
            self, fg_color=BACKGROUND, corner_radius=0
        )
        self.scroll.grid(
            row=1, column=1, sticky="nsew", padx=15, pady=(0, 12)
        )
        for column in range(2):
            self.scroll.grid_columnconfigure(column, weight=1)
        self.empty_label = ctk.CTkLabel(
            self.scroll,
            text="Nenhuma fonte aberta\n\nAdicione uma imagem, vídeo ou câmera pelo menu.",
            text_color=MUTED,
            font=ctk.CTkFont(size=16),
            justify="center",
        )
        self.empty_label.grid(
            row=0, column=0, columnspan=2, sticky="nsew", padx=25, pady=100
        )

        footer = ctk.CTkFrame(
            self, height=32, corner_radius=0, fg_color="#171A20"
        )
        footer.grid(row=2, column=1, sticky="ew")
        self.footer_status = ctk.CTkLabel(
            footer, text="Inicializando…", text_color=MUTED, anchor="w"
        )
        self.footer_status.pack(fill="x", padx=12, pady=6)

    def _load_models(self):
        self.footer_status.configure(text="Preparando os modelos de leitura…")
        threading.Thread(target=self._load_models_worker, daemon=True).start()

    def _load_models_worker(self):
        try:
            model11 = load_classifier(INTEGER_CLASSIFIER, 11)
            model100 = load_classifier(DECIMAL_CLASSIFIER, 100)
        except Exception as exc:
            self.schedule_ui(lambda error=exc: self._models_failed(error))
            return

        self.schedule_ui(lambda: self._models_ready(model11, model100))

    def schedule_ui(self, callback: Callable[[], None]):
        self.worker_callbacks.put(callback)

    def schedule_latest_frame(self, card: MeterCard, frame: np.ndarray):
        key = ("preview", card.card_id)
        self._schedule_coalesced(
            key, lambda image=frame: card._show_frame(image)
        )

    def schedule_latest_analysis(
        self,
        card: MeterCard,
        frame: np.ndarray,
        reading: str,
        status: str,
    ):
        key = ("analysis", card.card_id)
        self._schedule_coalesced(
            key,
            lambda image=frame, value=reading,
            message=status: card._render_analysis(
                image, value, message
            ),
        )

    def schedule_latest_status(self, card: MeterCard, callback: Callable[[], None]):
        self._schedule_coalesced(("status", card.card_id), callback)

    def _schedule_coalesced(self, key: tuple, callback: Callable[[], None]):
        with self.pending_ui_lock:
            self.pending_ui_callbacks[key] = callback
            if key not in self.queued_ui_keys:
                self.queued_ui_keys.add(key)
                self.worker_callbacks.put(key)

    def _poll_worker_callbacks(self):
        while True:
            try:
                item = self.worker_callbacks.get_nowait()
            except Empty:
                break
            if callable(item):
                callback = item
            else:
                with self.pending_ui_lock:
                    self.queued_ui_keys.discard(item)
                    callback = self.pending_ui_callbacks.pop(item, None)
            if callback is None:
                continue
            callback()
        if not self.closing:
            self.after(50, self._poll_worker_callbacks)

    def _models_ready(self, model11: dict, model100: dict):
        if self.closing:
            return
        self.model11 = model11
        self.model100 = model100
        for button in self.source_buttons:
            button.configure(state="normal")
        self.app_status.configure(text="Pronto para leitura", text_color=GREEN)
        self.footer_status.configure(
            text="Leitor pronto. Escolha uma imagem, vídeo ou câmera."
        )
        if self.migration_error:
            messagebox.showwarning(
                "Dados antigos",
                self.migration_error,
                parent=self,
            )

    def _models_failed(self, error: Exception):
        if self.closing:
            return
        self.app_status.configure(text="Não foi possível iniciar", text_color=RED)
        self.footer_status.configure(text="Confira os arquivos do aplicativo.")
        messagebox.showerror(
            "Erro ao iniciar o leitor",
            str(error),
            parent=self,
        )

    def _add_card(self, source: str | int, source_type: str, title: str):
        if self.model11 is None or self.model100 is None:
            return

        card = MeterCard(
            self.scroll,
            self,
            self.next_id,
            source,
            source_type,
            title,
        )
        self.next_id += 1
        if not self.cards:
            self.empty_label.grid_remove()
        self.cards.append(card)
        self._relayout()
        self._update_counter()
        self.footer_status.configure(text=f"Fonte adicionada: {title}")

    def add_video(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Selecionar vídeo",
            filetypes=[
                ("Arquivos de vídeo", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if path:
            self._add_card(path, "video", Path(path).name)

    def add_image(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Selecionar imagem",
            filetypes=[
                ("Arquivos de imagem", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if path:
            self._add_card(path, "image", Path(path).name)

    def add_camera(self):
        window = ctk.CTkToplevel(self)
        window.title("Selecionar câmera")
        window.geometry("390x210")
        window.transient(self)
        window.grab_set()
        window.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            window,
            text="Selecione uma câmera disponível",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=18, pady=(18, 8), sticky="w")
        status = ctk.CTkLabel(
            window, text="Procurando câmeras…", text_color=MUTED, anchor="w"
        )
        status.grid(row=1, column=0, padx=18, pady=5, sticky="ew")
        camera_list = ctk.CTkComboBox(window, values=[], state="disabled")
        camera_list.grid(row=2, column=0, padx=18, pady=5, sticky="ew")
        actions = ctk.CTkFrame(window, fg_color="transparent")
        actions.grid(row=3, column=0, padx=14, pady=12, sticky="e")
        ctk.CTkButton(
            actions,
            text="Procurar novamente",
            fg_color="#414752",
            command=lambda: self._scan_cameras(window, status, camera_list),
        ).pack(side="left", padx=4)
        ctk.CTkButton(
            actions,
            text="Abrir câmera",
            command=lambda: self._open_selected_camera(
                window, camera_list, status
            ),
        ).pack(side="left", padx=4)
        self._scan_cameras(window, status, camera_list)

    def _scan_cameras(self, window, status, camera_list):
        if not window.winfo_exists():
            return
        status.configure(text="Procurando câmeras…")
        camera_list.configure(values=[], state="disabled")
        threading.Thread(
            target=self._scan_cameras_worker,
            args=(window, status, camera_list),
            daemon=True,
        ).start()

    def _scan_cameras_worker(self, window, status, camera_list):
        available = find_cameras()
        self.schedule_ui(
            lambda: self._cameras_found(window, status, camera_list, available)
        )

    @staticmethod
    def _cameras_found(window, status, camera_list, available: list[int]):
        if not window.winfo_exists():
            return
        if not available:
            status.configure(text="Nenhuma câmera encontrada.")
            camera_list.configure(values=[], state="disabled")
            return
        values = [f"Câmera {index}" for index in available]
        camera_list.configure(values=values, state="readonly")
        camera_list.set(values[0])
        status.configure(text=f"{len(values)} câmera(s) disponível(is).")

    def _open_selected_camera(self, window, camera_list, status):
        selection = camera_list.get()
        if not selection.startswith("Câmera "):
            status.configure(text="Selecione uma câmera disponível.")
            return
        index = int(selection.removeprefix("Câmera "))
        window.destroy()
        self._add_card(index, "camera", selection)

    def remove_card(self, card: MeterCard):
        if card in self.cards:
            self.cards.remove(card)
        card.destroy()
        self._relayout()
        self._update_counter()
        if not self.cards:
            self.empty_label.grid(
                row=0, column=0, columnspan=2, sticky="nsew", padx=25, pady=100
            )

    def _update_counter(self):
        count = len(self.cards)
        self.counter_label.configure(
            text=f"{count} fonte" if count == 1 else f"{count} fontes"
        )

    def _relayout(self):
        for index, card in enumerate(self.cards):
            card.grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=8,
                pady=8,
            )

    def on_close(self):
        self.closing = True
        for card in list(self.cards):
            card.close()
        self.destroy()


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    app = HydrometerApp()
    app.mainloop()
