import csv
import hashlib
import json
import queue
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from queue import Empty
import tkinter.filedialog as filedialog
from tkinter import messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from leitor_gas.settings import (
    BACKGROUND,
    BORDER,
    CALIBRATIONS_DIR,
    DECIMAL_CLASSIFIER,
    FRAME_INTERVAL_MS,
    GREEN,
    INTEGER_CLASSIFIER,
    MUTED,
    PANEL,
    RED,
    RESULTS_DIR,
    TEXT,
)
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
        self.backend = "Não calibrado"
        self.last_values = None
        self.last_reading = "—"
        self.initialized_reading = False
        self.calibrating = False
        self.calibration_attempted = False
        self.reading_error = False
        self.writer = None
        self.recording = False
        self.image_ref = None

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

        self.pause_button = ctk.CTkButton(
            actions,
            text="Ⅱ  Pausar",
            height=34,
            fg_color="#414752",
            hover_color="#555D6B",
            command=self.toggle_pause,
        )
        self.pause_button.grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        self.calibrate_button = ctk.CTkButton(
            actions,
            text="Calibrar novamente",
            height=34,
            command=self.calibrate,
        )
        self.calibrate_button.grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        self.record_button = ctk.CTkButton(
            actions,
            text="Gravar vídeo",
            height=34,
            fg_color="#B64B35",
            hover_color="#D45B42",
            command=self.toggle_recording,
        )
        self.record_button.grid(row=1, column=0, padx=3, pady=3, sticky="ew")
        self.save_button = ctk.CTkButton(
            actions,
            text="Salvar leitura",
            height=34,
            fg_color="#286B43",
            hover_color="#348A57",
            command=self.save_reading,
            state="disabled",
        )
        self.save_button.grid(row=1, column=1, padx=3, pady=3, sticky="ew")

        if self.source_type == "image":
            self.pause_button.configure(state="disabled")
            self.record_button.configure(state="disabled")

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
                self._process_frame(self.frame, initialize=True)
            else:
                self.after(250, self.calibrate)
            return

        video_source = 0 if self.source_type == "camera" else str(self.source)
        try:
            self.capture = cv2.VideoCapture(video_source)
        except cv2.error as exc:
            self._set_status("Não foi possível abrir esta fonte.", error=True)
            messagebox.showerror("Erro ao abrir fonte", str(exc), parent=self.app)
            return
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            self._set_status("Não foi possível abrir esta fonte.", error=True)
            return

        self.running = True
        self._set_status("Ao vivo", color=GREEN)
        self.after(30, self._next_frame)

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
        if not self.calibration_file.is_file():
            return

        try:
            data = json.loads(self.calibration_file.read_text(encoding="utf-8"))
            if (
                not isinstance(data, dict)
                or int(data["width"]) <= 0
                or int(data["height"]) <= 0
                or not isinstance(data["rois"], list)
                or not data["rois"]
            ):
                raise ValueError("O arquivo de calibração está incompleto.")

            for roi in data["rois"]:
                for coordinate in ("x1", "y1", "x2", "y2"):
                    float(roi[coordinate])

            self.calibration = data
            self.backend = data.get("backend", "Calibração salva")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._set_status(f"Não foi possível carregar a calibração: {exc}", error=True)

    def _next_frame(self):
        if self.closed or not self.running or self.capture is None:
            return
        if self.paused:
            self.after(100, self._next_frame)
            return

        ok, frame = self.capture.read()
        if not ok:
            if self.source_type == "video":
                self.running = False
                self._release_writer()
                self._set_status("Vídeo finalizado.")
                return
            self.after(100, self._next_frame)
            return

        self.frame = frame
        if self.calibration is None:
            self._show_frame(frame)
            if not self.calibrating and not self.calibration_attempted:
                self.calibrate()
        elif self._compatible(frame):
            if self.reading_error:
                self._show_frame(frame)
            else:
                self._process_frame(frame, initialize=not self.initialized_reading)
        else:
            self._show_frame(frame)
            self._set_status("A resolução mudou. Calibre novamente.")

        if self.recording and self.writer is not None:
            try:
                self.writer.write(frame)
            except cv2.error as exc:
                self._release_writer()
                self._set_status("A gravação foi interrompida.", error=True)
                messagebox.showerror("Erro ao gravar", str(exc), parent=self.app)

        self.after(FRAME_INTERVAL_MS, self._next_frame)

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
            self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
            self.calibration_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            self.app.schedule_ui(lambda error=exc: self._calibration_failed(error))
            return

        self.app.schedule_ui(lambda: self._calibration_finished(data))

    def _calibration_finished(self, data: dict):
        self.calibrating = False
        if self.closed:
            return

        self.calibrate_button.configure(state="normal")
        self.calibration = data
        self.backend = data["backend"]
        self.last_values = None
        self.initialized_reading = False
        self.reading_error = False
        self._process_frame(self.frame, initialize=True)

    def _calibration_failed(self, error: Exception):
        self.calibrating = False
        if self.closed:
            return

        self.calibrate_button.configure(state="normal")
        self._set_status("Não foi possível calibrar esta fonte.", error=True)
        messagebox.showerror("Erro na calibração", str(error), parent=self.app)

    def _process_frame(self, frame: np.ndarray, initialize: bool = False):
        try:
            self._process_frame_impl(frame, initialize)
        except Exception as exc:
            self.reading_error = True
            self._release_writer()
            self._set_status("Não foi possível processar esta imagem.", error=True)
            messagebox.showerror("Erro na leitura", str(exc), parent=self.app)

    def _process_frame_impl(self, frame: np.ndarray, initialize: bool = False):
        if not self.calibration or not self.app.model11 or not self.app.model100:
            self._show_frame(frame)
            return
        if not self._compatible(frame):
            self._show_frame(frame)
            return

        height, width = frame.shape[:2]
        rois = self.calibration["rois"]
        if self.last_values is None or len(self.last_values) != len(rois):
            self.last_values = [None] * len(rois)

        last_roi = rois[-1]
        x1, y1, x2, y2 = self._clip_roi(last_roi, width, height)
        if x2 <= x1 or y2 <= y1:
            self._show_frame(frame)
            return

        last_class, last_score = classify_roi(
            frame[y1:y2, x1:x2], self.app.model100
        )
        last_value = round(last_class / 10.0, 1)
        self.last_values[-1] = {
            "kind": "decimal",
            "class": last_class,
            "value": last_value,
            "score": last_score,
        }

        update_integer_digits = initialize or last_value < 1.0
        if update_integer_digits:
            for index, roi in enumerate(rois[:-1]):
                x1, y1, x2, y2 = self._clip_roi(roi, width, height)
                if x2 <= x1 or y2 <= y1:
                    continue
                class_index, score = classify_roi(
                    frame[y1:y2, x1:x2], self.app.model11
                )
                value = None if class_index == 10 else class_index
                self.last_values[index] = {
                    "kind": "integer",
                    "class": class_index,
                    "value": value,
                    "score": score,
                }

        annotated = frame.copy()
        for roi, value in zip(rois, self.last_values):
            x1, y1, x2, y2 = self._clip_roi(roi, width, height)
            if value is None:
                label, color = "?", (0, 180, 255)
            elif value["value"] is None:
                label, color = "N", (0, 180, 255)
            elif value["kind"] == "decimal":
                label, color = f"{value['value']:.1f}", (50, 220, 100)
            else:
                label, color = str(value["value"]), (50, 220, 100)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

        reading_is_complete = all(
            value is not None and value["value"] is not None
            for value in self.last_values
        )
        if reading_is_complete:
            integer_part = "".join(
                str(value["value"]) for value in self.last_values[:-1]
            )
            self.last_reading = (
                integer_part + f"{self.last_values[-1]['value']:.1f}"
            )
            self.reading_label.configure(text=self.last_reading)
            self.save_button.configure(state="normal")
        else:
            self.last_reading = "—"
            self.reading_label.configure(text=self.last_reading)
            self.save_button.configure(state="disabled")

        self._show_frame(annotated)
        if initialize:
            status = "Leitura inicial concluída."
        elif update_integer_digits:
            status = "Leitura atualizada."
        else:
            status = "Acompanhando o último dígito."
        self._set_status(status, color=GREEN)
        if initialize:
            self.initialized_reading = True

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
        if self.source_type == "image":
            return

        self.paused = not self.paused
        self.pause_button.configure(text="▶  Retomar" if self.paused else "Ⅱ  Pausar")
        self.status_label.configure(
            text="●  Pausado" if self.paused else "●  Ao vivo",
            text_color=MUTED if self.paused else GREEN,
        )

    def toggle_recording(self):
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
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
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

        self.writer = writer
        self.recording = True
        self.record_button.configure(text="Parar gravação", fg_color=RED)
        self.info_label.configure(text=f"Gravando: {output_path.name}")

    def _release_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.recording = False
        if not self.closed:
            self.record_button.configure(text="Gravar vídeo", fg_color="#B64B35")

    def save_reading(self):
        if self.last_reading == "—":
            return

        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            output_path = RESULTS_DIR / "leituras.csv"
            is_new_file = not output_path.exists()
            with output_path.open("a", newline="", encoding="utf-8-sig") as file:
                writer = csv.writer(file, delimiter=";")
                if is_new_file:
                    writer.writerow(
                        ["data_hora", "fonte", "leitura", "backend", "rois"]
                    )
                writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        self.title_text,
                        self.last_reading,
                        self.backend,
                        len(self.calibration["rois"]),
                    ]
                )
        except OSError as exc:
            messagebox.showerror(
                "Erro ao salvar leitura", str(exc), parent=self.app
            )
            return

        self.info_label.configure(text="Leitura salva no histórico.")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.running = False
        self._release_writer()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.app.remove_card(self)


class HydrometerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Leitor de hidrômetros")
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

    def _poll_worker_callbacks(self):
        while True:
            try:
                callback = self.worker_callbacks.get_nowait()
            except Empty:
                break
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
        self._add_card(0, "camera", "Câmera local")

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
