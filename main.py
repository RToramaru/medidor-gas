import csv
import json
import hashlib
import threading
from datetime import datetime
from pathlib import Path
import tkinter.filedialog as filedialog
from tkinter import messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ai_edge_litert.interpreter import Interpreter

# ============================================================
# CONFIGURAÇÃO
# ============================================================
BASE = Path(__file__).resolve().parent
MODELOS = BASE / "modelos"
RESULTADOS = BASE / "resultados"
CALIBRACOES = RESULTADOS / "calibracoes"

YOLO_PATH = MODELOS / "stage1_meter_window_digits.pt"
YOLO_OV_PATH = MODELOS / "stage1_meter_window_digits_openvino_model"
CLASS11_PATH = MODELOS / "dig-class11_1910_s2_q.tflite"
CLASS100_PATH = MODELOS / "dig-class100-0182-s2_q.tflite"

CONFIANCA_YOLO = 0.15
IMG_SIZE_YOLO = 640
IOU_MAXIMO = 0.30
NORMALIZAR_FLOAT_255 = False
INTERVALO_VIDEO_MS = 35

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Paleta próxima ao painel escuro do projeto
BG = "#111318"
PANEL = "#1A1D24"
BORDER = "#303641"
GREEN = "#36C275"
RED = "#E05252"
TEXT = "#F2F4F8"
MUTED = "#9AA4B2"


def carregar_tflite(caminho: Path, classes_esperadas: int):
    if not caminho.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {caminho}")
    interpreter = Interpreter(model_path=str(caminho))
    interpreter.allocate_tensors()
    entrada = interpreter.get_input_details()[0]
    saida = interpreter.get_output_details()[0]
    formato = entrada["shape"].tolist()
    if len(formato) != 4 or int(formato[-1]) != 3:
        raise ValueError(f"Formato de entrada inesperado em {caminho.name}: {formato}")
    n_saida = int(np.prod(saida["shape"][1:]))
    if n_saida != classes_esperadas:
        raise ValueError(f"{caminho.name}: saída {n_saida}; esperadas {classes_esperadas} classes")
    return {
        "interpreter": interpreter,
        "entrada": entrada,
        "saida": saida,
        "altura": int(formato[1]),
        "largura": int(formato[2]),
        "classes": classes_esperadas,
    }


def preparar_tensor(roi, modelo):
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb).resize(
        (modelo["largura"], modelo["altura"]), Image.Resampling.BILINEAR
    )
    pixels = np.asarray(img, dtype=np.float32)
    entrada = modelo["entrada"]
    dtype = entrada["dtype"]
    if dtype == np.float32:
        if NORMALIZAR_FLOAT_255:
            pixels /= 255.0
        tensor = pixels.astype(np.float32)
    elif np.issubdtype(dtype, np.integer):
        escala, zero_point = entrada["quantization"]
        valores = pixels / 255.0 if 0 < escala < 0.01 else pixels
        if escala > 0:
            valores = np.round(valores / escala + zero_point)
        limites = np.iinfo(dtype)
        tensor = np.clip(valores, limites.min, limites.max).astype(dtype)
    else:
        raise TypeError(f"Tipo TFLite não suportado: {dtype}")
    return np.expand_dims(tensor, axis=0)


def classificar_roi(roi, modelo):
    interpreter = modelo["interpreter"]
    interpreter.set_tensor(modelo["entrada"]["index"], preparar_tensor(roi, modelo))
    interpreter.invoke()
    valores = np.asarray(interpreter.get_tensor(modelo["saida"]["index"])[0]).flatten()
    if np.issubdtype(valores.dtype, np.integer):
        escala, zero_point = modelo["saida"]["quantization"]
        valores = ((valores.astype(np.float32) - zero_point) * escala
                   if escala > 0 else valores.astype(np.float32))
    classe = int(np.argmax(valores))
    return classe, float(valores[classe])


def _tentar_yolo(imagem, caminho, device, backend):
    model = YOLO(str(caminho))
    results = model.predict(
        source=imagem, conf=CONFIANCA_YOLO, imgsz=IMG_SIZE_YOLO,
        device=device, verbose=False
    )
    if not results:
        raise RuntimeError("YOLO não retornou resultados")
    r = results[0]
    obb = getattr(r, "obb", None)
    boxes = getattr(r, "boxes", None)
    total = len(obb) if obb is not None else (len(boxes) if boxes is not None else 0)
    if total == 0:
        raise RuntimeError("Nenhuma detecção encontrada pelo YOLO")
    return results, model, backend


def executar_detector(imagem):
    erros = []
    try:
        import torch
        if torch.cuda.is_available():
            try:
                return _tentar_yolo(imagem, YOLO_PATH, 0, "CUDA / NVIDIA")
            except Exception as exc:
                erros.append(f"CUDA: {exc}")
    except Exception as exc:
        erros.append(f"CUDA indisponível: {exc}")

    if YOLO_OV_PATH.exists():
        try:
            import openvino as ov
            devices = ov.Core().available_devices
            if any(str(d).upper().startswith("GPU") for d in devices):
                try:
                    return _tentar_yolo(imagem, YOLO_OV_PATH, "intel:gpu", "OpenVINO / GPU")
                except Exception as exc:
                    erros.append(f"OpenVINO GPU: {exc}")
        except Exception as exc:
            erros.append(f"OpenVINO: {exc}")
        try:
            return _tentar_yolo(imagem, YOLO_OV_PATH, "intel:cpu", "OpenVINO / CPU")
        except Exception as exc:
            erros.append(f"OpenVINO CPU: {exc}")

    try:
        return _tentar_yolo(imagem, YOLO_PATH, "cpu", "PyTorch / CPU")
    except Exception as exc:
        erros.append(f"PyTorch CPU: {exc}")
    raise RuntimeError("Não foi possível executar o detector:\n\n" + "\n".join(erros))


def obter_deteccoes(resultado):
    obb = getattr(resultado, "obb", None)
    if obb is not None and len(obb):
        dados = zip(obb.xyxy.cpu().tolist(), obb.conf.cpu().tolist(), obb.cls.cpu().tolist())
    else:
        boxes = getattr(resultado, "boxes", None)
        if boxes is None or not len(boxes):
            return []
        dados = zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())
    saida = []
    for xyxy, conf, cls in dados:
        x1, y1, x2, y2 = map(float, xyxy)
        saida.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "conf": float(conf), "cls": int(cls)})
    return saida


def area(c):
    return max(0, c["x2"] - c["x1"]) * max(0, c["y2"] - c["y1"])


def iou(a, b):
    x1, y1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    x2, y2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def filtrar_sobreposicao(caixas):
    mantidas = []
    for c in sorted(caixas, key=lambda x: x["conf"], reverse=True):
        if not any(iou(c, outra) > IOU_MAXIMO for outra in mantidas):
            mantidas.append(c)
    return sorted(mantidas, key=lambda c: (c["x1"] + c["x2"]) / 2)


def detectar_posicoes(imagem):
    resultados, modelo, backend = executar_detector(imagem)
    deteccoes = obter_deteccoes(resultados[0])
    digitos, janelas = [], []
    for c in deteccoes:
        nome = str(modelo.names[c["cls"]]).strip().lower()
        c["nome"] = nome
        if "window" in nome:
            janelas.append(c)
        elif nome in {str(i) for i in range(10)} or nome == "u":
            digitos.append(c)

    # Se o YOLO localizar a janela, ignora dígitos fora dela.
    if janelas:
        janela = max(janelas, key=area)
        digitos = [c for c in digitos if
                   c["x2"] > janela["x1"] and c["x1"] < janela["x2"] and
                   c["y2"] > janela["y1"] and c["y1"] < janela["y2"]]
    digitos = filtrar_sobreposicao(digitos)
    if not digitos:
        raise RuntimeError("Nenhum ROI de dígito encontrado. Verifique o modelo e a imagem.")
    return digitos, backend


class MeterCard(ctk.CTkFrame):
    def __init__(self, master, app, card_id, source, source_type, title):
        super().__init__(master, fg_color=PANEL, corner_radius=14, border_width=1, border_color=BORDER)
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
        self.writer = None
        self.recording = False
        self.image_ref = None
        source_key = f"{source_type}:{source}"
        source_hash = hashlib.sha256(source_key.encode("utf-8")).hexdigest()[:16]
        self.calibration_file = CALIBRACOES / f"calibracao_{source_hash}.json"

        self.grid_columnconfigure(0, weight=1)
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 5))
        head.grid_columnconfigure(0, weight=1)
        self.title_label = ctk.CTkLabel(head, text=title, font=ctk.CTkFont(size=15, weight="bold"), anchor="w")
        self.title_label.grid(row=0, column=0, sticky="ew")
        self.status_dot = ctk.CTkLabel(head, text="●", text_color=GREEN, width=18)
        self.status_dot.grid(row=0, column=1, padx=4)
        ctk.CTkButton(head, text="×", width=30, height=28, fg_color="#353A44", hover_color=RED,
                      command=self.close).grid(row=0, column=2, padx=(4, 0))

        self.preview = ctk.CTkLabel(self, text="Abra uma fonte para visualizar", fg_color="#080A0E",
                                    corner_radius=9, height=235, text_color=MUTED)
        self.preview.grid(row=1, column=0, sticky="nsew", padx=12, pady=5)

        reading_box = ctk.CTkFrame(self, fg_color="#10251B", corner_radius=10)
        reading_box.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        reading_box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(reading_box, text="LEITURA ATUAL", text_color="#8EB99C",
                     font=ctk.CTkFont(size=10, weight="bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))
        self.reading_label = ctk.CTkLabel(reading_box, text="—", text_color="#55E58A",
                                          font=ctk.CTkFont(size=28, weight="bold"))
        self.reading_label.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))

        self.info_label = ctk.CTkLabel(self, text="Aguardando calibração", text_color=MUTED,
                                       anchor="w", justify="left", wraplength=360)
        self.info_label.grid(row=3, column=0, sticky="ew", padx=12, pady=(2, 7))

        actions = ctk.CTkFrame(self, fg_color="transparent")
        actions.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 12))
        for i in range(4):
            actions.grid_columnconfigure(i, weight=1)
        ctk.CTkButton(actions, text="Ⅱ Pausar", height=32, fg_color="#414752", hover_color="#555D6B",
                      command=self.toggle_pause).grid(row=0, column=0, padx=2, sticky="ew")
        ctk.CTkButton(actions, text="Calibrar", height=32, command=self.calibrate).grid(row=0, column=1, padx=2, sticky="ew")
        self.record_button = ctk.CTkButton(actions, text="Gravar", height=32, fg_color="#B64B35",
                                           hover_color="#D45B42", command=self.toggle_recording)
        self.record_button.grid(row=0, column=2, padx=2, sticky="ew")
        ctk.CTkButton(actions, text="Salvar leitura", height=32, fg_color="#286B43",
                      hover_color="#348A57", command=self.save_reading).grid(row=0, column=3, padx=2, sticky="ew")

        self._load_calibration()
        if source_type == "image":
            self.frame = cv2.imread(str(source))
            if self.frame is None:
                self.info_label.configure(text="Não foi possível abrir a imagem.")
            else:
                self._show_frame(self.frame)
                if self.calibration is not None:
                    # A calibração salva evita executar o YOLO novamente.
                    self.process_frame(self.frame, initialize=True)
                    self.initialized_reading = True
                else:
                    # Primeira execução: calibra e faz a leitura inicial automaticamente.
                    self.after(250, self.calibrate)
        else:
            self.capture = cv2.VideoCapture(0 if source_type == "camera" else str(source))
            if not self.capture.isOpened():
                self.info_label.configure(text="Não foi possível abrir a fonte de vídeo.")
                self.status_dot.configure(text_color=RED)
                return
            self.running = True
            self.after(30, self._next_frame)

    def _load_calibration(self):
        if not self.calibration_file.exists():
            return
        try:
            self.calibration = json.loads(self.calibration_file.read_text(encoding="utf-8"))
            self.backend = self.calibration.get("backend", "Calibração salva")
            self.info_label.configure(text="Calibração pronta. Iniciando leitura…")
        except Exception as exc:
            self.calibration = None
            self.info_label.configure(text=f"Falha ao carregar calibração: {exc}")

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
                self.status_dot.configure(text_color=MUTED)
                self.info_label.configure(text="Vídeo finalizado.")
                self._release_writer()
                return
            self.after(100, self._next_frame)
            return
        self.frame = frame
        if self.calibration is not None:
            if self._compatible(frame):
                # Na primeira leitura após abrir a fonte, inicializa todos os dígitos.
                inicializar = not self.initialized_reading
                self.process_frame(frame, initialize=inicializar)
                if inicializar:
                    self.initialized_reading = True
            else:
                self.info_label.configure(text="Resolução diferente da calibração. Clique em Calibrar.")
                self._show_frame(frame)
        else:
            self._show_frame(frame)
            self.info_label.configure(text="Calibrando automaticamente a primeira leitura…")
            if not self.calibrating:
                self.after(0, self.calibrate)
        if self.recording and self.writer is not None:
            try:
                self.writer.write(frame)
            except Exception as exc:
                self._release_writer()
                self.info_label.configure(text="Gravação interrompida.")
                messagebox.showerror("Erro ao gravar", str(exc))
        self.after(INTERVALO_VIDEO_MS, self._next_frame)

    def _compatible(self, frame):
        if not self.calibration:
            return False
        h, w = frame.shape[:2]
        return int(self.calibration.get("width", w)) == w and int(self.calibration.get("height", h)) == h

    def calibrate(self):
        if self.frame is None:
            messagebox.showwarning("Sem imagem", "Ainda não há um frame disponível.")
            return
        if getattr(self, "calibrating", False):
            return
        self.calibrating = True
        self.info_label.configure(text="Localizando os dígitos…")
        frame = self.frame.copy()
        threading.Thread(target=self._calibrate_worker, args=(frame,), daemon=True).start()

    def _calibrate_worker(self, frame):
        try:
            boxes, backend = detectar_posicoes(frame)
            h, w = frame.shape[:2]
            # Coordenadas normalizadas permitem restaurar a calibração em frames da mesma resolução
            # após reiniciar a aplicação. A compatibilidade de resolução ainda é validada.
            rois = []
            for c in boxes:
                rois.append({
                    "x1": int(c["x1"]), "y1": int(c["y1"]),
                    "x2": int(c["x2"]), "y2": int(c["y2"]),
                    "conf": c["conf"], "nome": c.get("nome", "")
                })
            data = {"width": w, "height": h, "backend": backend, "rois": rois}
            self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
            self.calibration_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            self.after(0, lambda: self._calibration_done(data, backend))
        except Exception as exc:
            error = str(exc)
            self.after(0, lambda error=error: self._calibration_error(error))

    def _calibration_done(self, data, backend):
        self.calibrating = False
        if self.closed:
            return
        self.calibration = data
        self.backend = backend
        self.info_label.configure(text="Calibração concluída.")
        if self.frame is not None:
            self.last_values = None
            self.initialized_reading = False
            self.process_frame(self.frame, initialize=True)
            self.initialized_reading = True

    def _calibration_error(self, error):
        self.calibrating = False
        if not self.closed:
            self.info_label.configure(text="Erro na calibração.")
            messagebox.showerror("Erro ao calibrar", error)

    def process_frame(self, frame, initialize=False):
        if not self.calibration or self.app.model11 is None or self.app.model100 is None:
            self._show_frame(frame)
            return
        if not self._compatible(frame):
            self._show_frame(frame)
            return

        h, w = frame.shape[:2]
        rois = self.calibration["rois"]
        if self.last_values is None or len(self.last_values) != len(rois):
            self.last_values = [None] * len(rois)

        # Último ROI sempre é classificado pelo modelo de 100 classes.
        last_roi = rois[-1]
        x1, y1, x2, y2 = self._clip_roi(last_roi, w, h)
        if x2 <= x1 or y2 <= y1:
            self._show_frame(frame)
            return
        last_class, last_score = classificar_roi(frame[y1:y2, x1:x2], self.app.model100)
        last_value = round(last_class / 10.0, 1)
        self.last_values[-1] = {"kind": "100", "class": last_class, "value": last_value, "score": last_score}

        # Na leitura inicial/calibração, classifica todos os dígitos uma vez.
        # Depois, atualiza os dígitos anteriores somente quando o último é 0.x,
        # em que x pode ser qualquer dígito de 0 a 9 (0.0 até 0.9).
        atualizar_anteriores = initialize or (0.0 <= last_value < 1.0)
        if atualizar_anteriores:
            for i, box in enumerate(rois[:-1]):
                a, b, c, d = self._clip_roi(box, w, h)
                if c <= a or d <= b:
                    continue
                classe, score = classificar_roi(frame[b:d, a:c], self.app.model11)
                value = None if classe == 10 else classe
                self.last_values[i] = {"kind": "11", "class": classe, "value": value, "score": score}
        annotated = frame.copy()
        for i, box in enumerate(rois):
            a, b, c, d = self._clip_roi(box, w, h)
            item = self.last_values[i]
            if item is None:
                label = "?"
                color = (0, 180, 255)
            elif item["value"] is None:
                label = "N"
                color = (0, 180, 255)
            elif item["kind"] == "100":
                label = f"{item['value']:.1f}"
                color = (50, 220, 100)
            else:
                label = str(item["value"])
                color = (50, 220, 100)
            cv2.rectangle(annotated, (a, b), (c, d), color, 2)
            cv2.putText(annotated, label, (a, max(20, b - 7)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2, cv2.LINE_AA)

        # Só monta a leitura completa se todos os dígitos anteriores já foram reconhecidos.
        if all(v is not None and v["value"] is not None for v in self.last_values):
            prefix = "".join(str(v["value"]) for v in self.last_values[:-1])
            self.last_reading = prefix + f"{self.last_values[-1]['value']:.1f}"
            self.reading_label.configure(text=self.last_reading)
        else:
            self.last_reading = "—"
            self.reading_label.configure(text="—")

        self._show_frame(annotated)
        if initialize:
            extra = "Leitura inicial: todos os dígitos classificados."
        elif atualizar_anteriores:
            extra = "Último dígito entre 0.0 e 0.9: dígitos anteriores atualizados."
        else:
            extra = "Último dígito atualizado; dígitos anteriores mantidos."
        self.info_label.configure(text=extra)

    @staticmethod
    def _clip_roi(box, w, h):
        x1 = max(0, min(w - 1, int(box["x1"])))
        y1 = max(0, min(h - 1, int(box["y1"])))
        x2 = max(0, min(w, int(box["x2"])))
        y2 = max(0, min(h, int(box["y2"])))
        return x1, y1, x2, y2

    def _show_frame(self, frame):
        if self.closed or frame is None:
            return
        h, w = frame.shape[:2]
        max_w, max_h = 540, 300
        scale = min(max_w / w, max_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = cv2.cvtColor(cv2.resize(frame, (nw, nh)), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self.image_ref = ctk.CTkImage(light_image=pil, dark_image=pil, size=(nw, nh))
        self.preview.configure(image=self.image_ref, text="")

    def toggle_pause(self):
        if self.source_type == "image":
            return
        self.paused = not self.paused
        self.status_dot.configure(text_color=MUTED if self.paused else GREEN)

    def toggle_recording(self):
        if self.source_type == "image":
            messagebox.showinfo("Gravação", "A gravação de vídeo está disponível para fontes de vídeo e câmera.")
            return
        if not self.recording:
            if self.frame is None:
                return
            RESULTADOS.mkdir(parents=True, exist_ok=True)
            path = RESULTADOS / f"video_{self.card_id}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            h, w = self.frame.shape[:2]
            fps = 25.0
            if self.capture is not None:
                fps_lido = self.capture.get(cv2.CAP_PROP_FPS)
                if fps_lido and 1 <= fps_lido <= 120:
                    fps = fps_lido
            self.writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if not self.writer.isOpened():
                self.writer = None
                messagebox.showerror("Gravação", "Não foi possível iniciar a gravação do vídeo.")
                return
            self.recording = True
            self.record_button.configure(text="Parar gravação", fg_color=RED)
            self.info_label.configure(text=f"Gravando em {path.name}")
        else:
            self._release_writer()
            self.record_button.configure(text="Gravar", fg_color="#B64B35")

    def _release_writer(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.recording = False
        if hasattr(self, "record_button") and not self.closed:
            self.record_button.configure(text="Gravar", fg_color="#B64B35")

    def save_reading(self):
        RESULTADOS.mkdir(parents=True, exist_ok=True)
        path = RESULTADOS / "leituras.csv"
        novo = not path.exists()
        with path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter=";")
            if novo:
                writer.writerow(["data_hora", "fonte", "leitura", "backend", "rois"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), self.title_text,
                             self.last_reading, self.backend,
                             len(self.calibration["rois"]) if self.calibration else 0])
        self.info_label.configure(text=f"Leitura salva em {path.name}")

    def close(self):
        self.closed = True
        self.running = False
        self._release_writer()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.app.remove_card(self)


class HidrometroApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hidrômetro · Monitoramento Inteligente")
        self.geometry("1450x940")
        self.minsize(1100, 720)
        self.configure(fg_color=BG)
        RESULTADOS.mkdir(parents=True, exist_ok=True)
        CALIBRACOES.mkdir(parents=True, exist_ok=True)

        self.cards = []
        self.next_id = 1
        self.model11 = None
        self.model100 = None
        self._build_ui()
        self.after(150, self._load_models)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#171A20")
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_propagate(False)
        ctk.CTkLabel(sidebar, text="◉  HIDROMETRO", font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=TEXT).pack(anchor="w", padx=18, pady=(25, 4))
        ctk.CTkLabel(sidebar, text="LEITURA DE HIDRÔMETROS", font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=MUTED).pack(anchor="w", padx=20, pady=(0, 24))
        ctk.CTkLabel(sidebar, text="FONTES", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=MUTED).pack(anchor="w", padx=18, pady=(4, 8))
        ctk.CTkButton(sidebar, text="＋  Abrir vídeo", anchor="w", command=self.add_video).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(sidebar, text="▣  Abrir imagem", anchor="w", fg_color="#303641", hover_color="#414957",
                      command=self.add_image).pack(fill="x", padx=12, pady=4)
        ctk.CTkButton(sidebar, text="◉  Câmera local", anchor="w", fg_color="#303641", hover_color="#414957",
                      command=self.add_camera).pack(fill="x", padx=12, pady=4)
        ctk.CTkFrame(sidebar, height=1, fg_color=BORDER).pack(fill="x", padx=14, pady=18)
        ctk.CTkLabel(sidebar, text="STATUS", font=ctk.CTkFont(size=11, weight="bold"),
                     text_color=MUTED).pack(anchor="w", padx=18, pady=(0, 8))
        self.app_status = ctk.CTkLabel(sidebar, text="Preparando…", justify="left", anchor="w",
                                       wraplength=185, text_color=MUTED)
        self.app_status.pack(fill="x", padx=18)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=1, sticky="ew", padx=22, pady=(20, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text="Painel de monitoramento", font=ctk.CTkFont(size=25, weight="bold"),
                     text_color=TEXT).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(header, text="Acompanhe a leitura em imagens, vídeos e câmera",
                     text_color=MUTED).grid(row=1, column=0, sticky="w", pady=(3, 0))
        self.counter_label = ctk.CTkLabel(header, text="0 fontes ativas", text_color="#9BC6FF",
                                          fg_color="#1D2A3B", corner_radius=8)
        self.counter_label.grid(row=0, column=1, rowspan=2, padx=(12, 0))

        self.scroll = ctk.CTkScrollableFrame(self, fg_color=BG, corner_radius=0)
        self.scroll.grid(row=1, column=1, sticky="nsew", padx=15, pady=(0, 15))
        self.scroll.grid_columnconfigure(0, weight=1)
        self.scroll.grid_columnconfigure(1, weight=1)
        self.empty_label = ctk.CTkLabel(self.scroll, text="Nenhuma fonte aberta\n\nAdicione um vídeo, imagem ou câmera pelo menu lateral.",
                                        text_color=MUTED, font=ctk.CTkFont(size=16), justify="center")
        self.empty_label.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=25, pady=100)

        footer = ctk.CTkFrame(self, height=28, corner_radius=0, fg_color="#171A20")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.footer_status = ctk.CTkLabel(footer, text="Preparando o aplicativo…", text_color=MUTED, anchor="w")
        self.footer_status.pack(fill="x", padx=12, pady=5)

    def _load_models(self):
        try:
            self.footer_status.configure(text="Preparando leitura…")
            self.model11 = carregar_tflite(CLASS11_PATH, 11)
            self.model100 = carregar_tflite(CLASS100_PATH, 100)
            self.app_status.configure(text="Pronto para leitura", text_color=GREEN)
            self.footer_status.configure(text="Aplicativo pronto. Abra uma fonte para começar.")
        except Exception as exc:
            self.app_status.configure(text="Não foi possível iniciar", text_color=RED)
            self.footer_status.configure(text="Falha ao iniciar. Consulte os detalhes do aviso.")
            messagebox.showerror("Erro ao carregar modelos", str(exc))

    def _add_card(self, source, source_type, title):
        card_id = self.next_id
        self.next_id += 1
        if len(self.cards) == 0:
            self.empty_label.grid_remove()
        card = MeterCard(self.scroll, self, card_id, source, source_type, title)
        self.cards.append(card)
        self._relayout()
        self.counter_label.configure(text=f"{len(self.cards)} fonte(s) ativa(s)")
        self.footer_status.configure(text=f"Fonte adicionada: {title}")
        return card

    def add_video(self):
        path = filedialog.askopenfilename(title="Selecionar vídeo", filetypes=[
            ("Vídeos", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("Todos os arquivos", "*.*")])
        if path:
            self._add_card(path, "video", Path(path).name)

    def add_image(self):
        path = filedialog.askopenfilename(title="Selecionar imagem", filetypes=[
            ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos os arquivos", "*.*")])
        if path:
            self._add_card(path, "image", Path(path).name)

    def add_camera(self):
        self._add_card(0, "camera", "Câmera local")

    def remove_card(self, card):
        if card in self.cards:
            self.cards.remove(card)
        card.destroy()
        self._relayout()
        self.counter_label.configure(text=f"{len(self.cards)} fonte(s) ativa(s)")
        if not self.cards:
            self.empty_label.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=25, pady=100)

    def _relayout(self):
        for i, card in enumerate(self.cards):
            card.grid(row=i // 2, column=i % 2, sticky="nsew", padx=8, pady=8)

    def on_close(self):
        for card in list(self.cards):
            card.close()
        self.destroy()


def main():
    app = HidrometroApp()
    app.mainloop()


if __name__ == "__main__":
    main()
