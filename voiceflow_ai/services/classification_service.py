import asyncio
import collections
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from voiceflow_ai.core.config import settings as c
from voiceflow_ai.core.logger import get_logger

logger = get_logger("ClassificationService")


class ClassificationService:
    def __init__(self):
        self.distil_model = None
        self.tokenizer = None
        self.distil_model_medicare = None
        self.tokenizer_medicare = None
        self.distil_model_medicare_b = None
        self.tokenizer_medicare_b = None
        self.distil_model_aca = None
        self.tokenizer_aca = None
        self.distil_model_aca_b = None
        self.tokenizer_aca_b = None
        self.distil_model_fe = None
        self.tokenizer_fe = None
        self.distil_model_fe_b = None
        self.tokenizer_fe_b = None
        self.shutdown_in_progress = False
        self.active_classifications_count = 0
        self.active_classifications = collections.deque()

    def initialize_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Initialize DistilBerts
            model_path = c.DISTIL_MODEL
            model_path_medicare_b = c.MEDICARE_MODEL_B
            model_path_medicare_11 = c.MEDICARE_MODEL_11
            model_path_medicare_12 = c.MEDICARE_MODEL_12
            model_path_aca = c.ACA_MODEL
            model_path_aca_b = c.ACA_MODEL_B
            model_path_fe_b = c.FE_MODEL_B

            self.distil_model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            )
            self.distil_model.to(device)  # Move the model to the GPU
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # self.distil_model_medicare = AutoModelForSequenceClassification.from_pretrained(model_path_medicare)
            # self.distil_model_medicare.to(device)  # Move the model to the GPU
            # self.tokenizer_medicare = AutoTokenizer.from_pretrained(model_path_medicare)

            self.distil_model_medicare_b = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_path_medicare_b
                )
            )
            self.distil_model_medicare_b.to(device)  # Move the model to the GPU
            self.tokenizer_medicare_b = AutoTokenizer.from_pretrained(
                model_path_medicare_b
            )

            self.distil_model_medicare_11 = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_path_medicare_11
                )
            )
            self.distil_model_medicare_11.to(device)  # Move the model to the GPU
            self.tokenizer_medicare_11 = AutoTokenizer.from_pretrained(
                model_path_medicare_11
            )

            self.distil_model_medicare_12 = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_path_medicare_12
                )
            )
            self.distil_model_medicare_12.to(device)  # Move the model to the GPU
            self.tokenizer_medicare_12 = AutoTokenizer.from_pretrained(
                model_path_medicare_12
            )

            self.distil_model_aca = AutoModelForSequenceClassification.from_pretrained(
                model_path_aca
            )
            self.distil_model_aca.to(device)  # Move the model to the GPU
            self.tokenizer_aca = AutoTokenizer.from_pretrained(model_path_aca)

            self.distil_model_aca_b = (
                AutoModelForSequenceClassification.from_pretrained(model_path_aca_b)
            )
            self.distil_model_aca_b.to(device)  # Move the model to the GPU
            self.tokenizer_aca_b = AutoTokenizer.from_pretrained(model_path_aca_b)

            # self.distil_model_fe = AutoModelForSequenceClassification.from_pretrained(model_path_fe)
            # self.distil_model_fe.to(device)  # Move the model to the GPU
            # self.tokenizer_fe = AutoTokenizer.from_pretrained(model_path_fe)

            self.distil_model_fe_b = AutoModelForSequenceClassification.from_pretrained(
                model_path_fe_b
            )
            self.distil_model_fe_b.to(device)  # Move the model to the GPU
            self.tokenizer_fe_b = AutoTokenizer.from_pretrained(model_path_fe_b)

            logger.info(f"Distilberts loaded and device is: {device}")
        except Exception as e:
            logger.error(f"Error during whisper initialization: {e}", exc_info=True)
            raise

    def classify_audio(self, transcribed_text, connection_id, model_type, call_type):
        self.active_classifications_count += 1
        self.active_classifications.append(time.time())

        # Default model and tokenizer
        model = self.distil_model
        tokenizer = self.tokenizer
        model_used = "A"
        if call_type == "medicare":
            if model_type == "A":
                model_used = "mc_10.3"
            elif model_type == "B":
                model = self.distil_model_medicare
                tokenizer = self.tokenizer_medicare
                model_used = "mc2_3.3"
            elif model_type == "C":
                model = self.distil_model_medicare_b
                tokenizer = self.tokenizer_medicare_b
                model_used = "mc2_8.3"
            elif model_type == "11":
                model = self.distil_model_medicare_11
                tokenizer = self.tokenizer_medicare_11
                model_used = "medicare11"
            elif model_type == "12":
                model = self.distil_model_medicare_12
                tokenizer = self.tokenizer_medicare_12
                model_used = "medicare12"
            else:
                model_used = "E-medicare"

        elif call_type == "aca":
            model = self.distil_model_aca
            tokenizer = self.tokenizer_aca
            if model_type == "A":
                model_used = "aca_5.3"
            elif model_type == "B":
                model = self.distil_model_aca_b
                tokenizer = self.tokenizer_aca_b
                model_used = "aca2_3.2"
            else:
                model_used = "E-aca"

        elif call_type == "fe":
            model = self.distil_model_fe_b
            tokenizer = self.tokenizer_fe_b
            if model_type == "A":
                model_used = "A-fe"
            elif model_type == "B":
                model = self.distil_model_fe_b
                tokenizer = self.tokenizer_fe_b
                model_used = "B-fe"
            else:
                model_used = "E-fe"

        if model is not None and tokenizer is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # In your classify_audio function
            inputs = tokenizer(
                transcribed_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = inputs.to(device)  # Move the inputs to the GPU
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities)

            # Detach the tensors before returning them
            predicted_class = predicted_class.detach()
            probabilities = probabilities.detach()

            label, confidence = self.determine_label(
                predicted_class, probabilities, model_type, call_type
            )

            self.active_classifications_count -= 1
            self.active_classifications.popleft()

            return label, confidence, model_used

    @staticmethod
    def determine_label(predicted_class, probabilities, model_type, call_type):
        predicted_class = predicted_class.item()  # Convert torch.Tensor to int
        label_confidence = probabilities[0][predicted_class].item()
        confidence_threshold = 0.65  # configurable parameter

        labels_33_classes = {
            0: "NQI",
            1: "AD",
            2: "AM",
            3: "WP",
            4: "AP",
            5: "BOT",
            6: "CB",
            7: "COM",
            8: "CONF",
            9: "DNC",
            10: "N",
            11: "N-",
            12: "NE",
            13: "GNI",
            14: "PN",
            15: "P",
            16: "PQ",
            17: "U",
            18: "TMC",
            19: "YJC",
            20: "AHA",
            21: "AHI",
            22: "PI",
            23: "JL",
            24: "PG",
            25: "NG",
            26: "PG+",
            27: "NQA",
            28: "NQW",
            29: "TC",
            30: "Q",
            31: "R",
            32: "LB",
        }

        # Define the labels for each class
        labels_18_classes = {
            0: "ABN",
            1: "AD",
            2: "AM",
            3: "ANI",
            4: "AP",
            5: "BOT",
            6: "CB",
            7: "COM",
            8: "CONF",
            9: "DNC",
            10: "N",
            11: "N-",
            12: "NE",
            13: "NI",
            14: "PN",
            15: "P",
            16: "PQ",
            17: "U",
        }
        labels_3_classes = {0: "non-sales", 1: "sales", 2: "neutral"}

        label_50_classes = {
            0: "NQD",
            1: "AD",
            2: "AM",
            3: "DP",
            4: "AP1",
            5: "BOT",
            6: "CB",
            7: "COM",
            8: "CONF",
            9: "DNC",
            10: "N",
            11: "N-",
            12: "NE",
            13: "GNI",
            14: "PN",
            15: "P",
            16: "PQ",
            17: "U",
            18: "TMC",
            19: "YJC",
            20: "AHFE",
            21: "AHI",
            22: "AHLI",
            23: "JL",
            24: "PG",
            25: "NG",
            26: "PG+",
            27: "NQA",
            28: "NQNH",
            29: "TC",
            30: "Q",
            31: "R",
            32: "LB",
            33: "AP2",
            34: "AP3",
            35: "AP4",
            36: "AP5",
            37: "AP6",
            38: "NAS",
            39: "WHO",
            40: "CASH",
            41: "WANT",
            42: "BENE",
            43: "OI",
            44: "SEC",
            45: "WHOM",
            46: "HOW",
            47: "DIE",
            48: "SCAM",
            49: "TELE",
        }

        labels_84_classes = {
            0: "NQD",
            1: "AD",
            2: "AM",
            3: "DP",
            4: "AP1",
            5: "BOT",
            6: "CB",
            7: "COM",
            8: "CONF",
            9: "DNC",
            10: "N",
            11: "N-",
            12: "NE",
            13: "GNI",
            14: "PN",
            15: "P",
            16: "PQ",
            17: "U",
            18: "TMC",
            19: "YJC",
            20: "AHM",
            21: "AHI",
            22: "AB",
            23: "JL",
            24: "PG",
            25: "NG",
            26: "PG+",
            27: "NQA",
            28: "NQNH",
            29: "TC",
            30: "Q",
            31: "R",
            32: "LB",
            33: "AP2",
            34: "AP3",
            35: "AP4",
            36: "AP5",
            37: "AP6",
            38: "ABCD",
            39: "WHO",
            40: "ADV",
            41: "WANT",
            42: "BENE",
            43: "AGNT",
            44: "SEC",
            45: "WHOM",
            46: "HOW",
            47: "COST",
            48: "SCAM",
            49: "TELE",
            50: "AP7",
            51: "ANI2",
            52: "ANI3",
            53: "ANI4",
            54: "ANI5",
            55: "ANI6",
            56: "ANI7",
            57: "DIS",
            58: "DVH",
            59: "ENRL",
            60: "ITM",
            61: "MAID",
            62: "MCARE",
            63: "MDCP",
            64: "MM",
            65: "MTC",
            66: "PNI",
            67: "QA",
            68: "SUPL",
            69: "UDK",
            70: "ANI1",
            71: "NQI",
            72: "ABNI",
            73: "B",
            74: "CE",
            75: "CQ",
            76: "HOLD",
            77: "HU",
            78: "PCOST",
            79: "PCQ",
            80: "PTIME",
            81: "TIME",
            82: "FD",
            83: "BDNC",
        }

        labels_75_classes = {
            0: "AB",
            1: "ABCD",
            2: "ABNI",
            3: "AD",
            4: "ADV",
            5: "AGNT",
            6: "AHI",
            7: "AHM",
            8: "AM",
            9: "ANI1",
            10: "ANI2",
            11: "ANI3",
            12: "ANI4",
            13: "ANI5",
            14: "ANI6",
            15: "ANI7",
            16: "AP1",
            17: "AP2",
            18: "AP3",
            19: "AP4",
            20: "AP5",
            21: "AP6",
            22: "AP7",
            23: "B",
            24: "BDNC",
            25: "BENE",
            26: "BN",
            27: "BOT",
            28: "CB",
            29: "CE",
            30: "COM",
            31: "CONF",
            32: "CQ",
            33: "DIS",
            34: "DNC",
            35: "DP",
            36: "DVH",
            37: "ENRL",
            38: "FD",
            39: "GNI",
            40: "HOLD",
            41: "HU",
            42: "ITM",
            43: "JL",
            44: "LB",
            45: "MAID",
            46: "MCARE",
            47: "MM",
            48: "MTC",
            49: "N",
            50: "N-",
            51: "NE",
            52: "NG",
            53: "NQA",
            54: "NQD",
            55: "NQI",
            56: "P",
            57: "PCOST",
            58: "PG",
            59: "PG+",
            60: "PN",
            61: "PNI",
            62: "PQ",
            63: "PTIME",
            64: "Q",
            65: "QA",
            66: "R",
            67: "SCAM",
            68: "SEC",
            69: "SUPL",
            70: "TC",
            71: "TELE",
            72: "TMC",
            73: "U",
            74: "YJC",
        }

        labels_50_classes_aca = {
            0: "ACA",
            1: "ACE",
            2: "AD",
            3: "AGNT",
            4: "AHA",
            5: "AHI",
            6: "AM",
            7: "AP",
            8: "B",
            9: "BDNC",
            10: "BENE",
            11: "BN",
            12: "BOT",
            13: "CB",
            14: "CE",
            15: "COM",
            16: "CONF",
            17: "COST",
            18: "CQ",
            19: "DNC",
            20: "ELI",
            21: "FD",
            22: "GNI",
            23: "HOLD",
            24: "ICE",
            25: "IHG",
            26: "JL",
            27: "LB",
            28: "N",
            29: "N-",
            30: "NE",
            31: "NG",
            32: "NQA",
            33: "NQI",
            34: "P",
            35: "PG",
            36: "PG+",
            37: "PN",
            38: "Q",
            39: "QP",
            40: "R",
            41: "SCAM",
            42: "SEC",
            43: "SUB",
            44: "TC",
            45: "TELE",
            46: "TIME",
            47: "TMC",
            48: "U",
            49: "YJC",
        }

        labels_25_classes = {
            0: "ABN",
            1: "AD",
            2: "AM",
            3: "ANI",
            4: "AP",
            5: "BOT",
            6: "CB",
            7: "CDC",
            8: "COM",
            9: "CONF",
            10: "DNC",
            11: "LB",
            12: "N",
            13: "N-",
            14: "NE",
            15: "NI",
            16: "NIC",
            17: "NIE",
            18: "NUM",
            19: "P",
            20: "PN",
            21: "PQ",
            22: "QR",
            23: "SC",
            24: "U",
        }

        label_27_classes = {
            0: "N",
            1: "N2",
            2: "N-",
            3: "U",
            4: "CB-M",
            5: "CB-S",
            6: "NI-A",
            7: "NI",
            8: "NI-REPEAT",
            9: "SCAM",
            10: "AI",
            11: "COM",
            12: "ABN",
            13: "AD",
            14: "AM",
            15: "DNC",
            16: "P",
            17: "PN",
            18: "NE-AGE",
            19: "NE",
            20: "PQ",
            21: "NI-AGE",
            22: "BC",
            23: "AP",
            24: "CONF-DNU",
            25: "CONF-DNK",
            26: "LB",
        }

        classification_label = None

        if c.TYPE:  # If c.TYPE is True, it means it has 8 classes
            if call_type == "medicare":
                labels = labels_18_classes
                classification_label = labels.get(predicted_class, "N")
                if model_type == "A":
                    labels = labels_18_classes
                    classification_label = labels.get(predicted_class, "N")
                elif model_type == "B":
                    labels = labels_84_classes
                    classification_label = labels.get(predicted_class, "N")
                elif model_type == "C":
                    labels = labels_75_classes
                    classification_label = labels.get(predicted_class, "N")
                elif model_type == "11":
                    labels = labels_25_classes
                    classification_label = labels.get(predicted_class, "N")
                elif model_type == "12":
                    labels = label_27_classes
                    classification_label = labels.get(predicted_class, "N")

            elif call_type == "aca":
                labels = labels_33_classes
                classification_label = labels.get(predicted_class, "N")
                if model_type == "A":
                    labels = labels_33_classes
                    classification_label = labels.get(predicted_class, "N")
                if model_type == "B":
                    labels = labels_50_classes_aca
                    classification_label = labels.get(predicted_class, "N")

            elif call_type == "fe":
                labels = label_50_classes
                classification_label = labels.get(predicted_class, "N")

        else:  # If c.TYPE is False, it means it has 3 classes
            labels = labels_3_classes
            if label_confidence < confidence_threshold:
                classification_label = "neutral"
            else:
                classification_label = labels.get(predicted_class, "neutral")

        return classification_label, label_confidence

    async def shutdown(self):
        self.shutdown_in_progress = True
        while self.active_classifications_count > 0:
            await asyncio.sleep(0.1)
        self.distil_model = None
        logger.info("Graceful shutdown completed.")
