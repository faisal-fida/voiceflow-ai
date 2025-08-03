import re
import string
import time
from itertools import product

import requests

from voiceflow_ai.core.config import settings as c
from voiceflow_ai.core.logger import get_logger

logger = get_logger("transcription_processor")


class TranscriptionProcessor:
    def __init__(self):
        contractions = c.CONTRACTIONS
        exact_search_dict = c.EXACT_SEARCH_DICT
        substring_search_dict = c.SUBSTRING_SEARCH_DICT
        self.contractions = contractions
        self.exact_search_dict = {
            k.lower(): v for v, k_list in exact_search_dict.items() for k in k_list
        }
        self.substring_search_dict = {
            k.lower(): v for v, k_list in substring_search_dict.items() for k in k_list
        }

    def contract_text(self, transcribed_text):
        # Function to replace full forms with contractions
        def replace(match):
            return self.contractions[match.group(0)]

        # Replace full forms with contractions in the transcribed text
        transcribed_text = re.sub(
            r"\b" + "|".join(self.contractions.keys()) + r"\b",
            replace,
            transcribed_text,
            flags=re.IGNORECASE,
        )

        return transcribed_text

    def process_transcription(self, transcribed_text, connection_id, model_type, call_type, turn):
        # Convert to lowercase
        transcribed_text = transcribed_text.lower()

        # Contract the text
        transcribed_text = self.contract_text(transcribed_text)

        punctuation = string.punctuation

        # Remove punctuation except apostrophe
        transcribed_text = "".join(
            ch for ch in transcribed_text if ch not in punctuation.replace("-", "") or ch == "'"
        ).lower()

        removal_list = [
            "background noise drowns out speaker",
            "drowned out by background noise",
            "knocking on the door",
            "knocking on door",
            "muffled radio static",
            "audio fades out",
            "soft piano music",
            "indistinct radio chatter",
            "sad trombone music",
            "audio cuts out",
            "soft music",
            "door opens",
            "upbeat music",
            "light music",
            "radio static",
            "static crackling",
            "audience applauds",
            "piano music",
            "electronic beeping",
            "ominous music",
            "keyboard clicking",
            "muffled speaking",
            "muffled speech",
            "dramatic music",
            "muffled talking",
            "audience applauding",
            "audience laughs",
            "dog barking",
            "dogs barking",
            "gentle music",
            "electronic music",
            "gun fires",
            "non-english speech",
            "air whooshing",
            "phone buzzes",
            "audience claps",
            "engine revving",
            "swoosh sound",
            "indistinct chatter",
            "fart noise",
            "clock ticking",
            "electronic jingle",
            "drum roll",
            "gavel bangs",
            "garbled speech",
            "electronic noise",
            "birds chirping",
            "thunder rumbling",
            "wind howling",
            "crowd cheering",
            "crowd chattering",
            "camera clicks",
            "kissing sound",
            "door slams",
            "bell dings",
            "audio out",
            "phone vibrating",
            "water gurgling",
            "baby babbling",
            "car horn",
            "keyboard clacking",
            "radio chatter",
            "muffled voices",
            "electronic sounds",
            "door slamming",
            "phone ringing",
            "audience laughing",
            "dog barks",
            "baby crying",
            "sirens blaring",
            "cat meows",
            "clears throat",
            "audience clapping",
            "whooshing",
            "static",
            "inaudible",
            "mumbling",
            "chuckling",
            "phone rings",
            "gunfire",
            "growls",
            "farting",
            "barking",
            "chuckles",
            "thud",
            "groaning",
            "growl",
            "music",
            "Coughing",
            "banging",
            "boop",
            "indiscernible",
            "sighs",
            "sigh",
            "sings",
            "coughs",
            "knocking",
            "pause",
            "cheering",
            "whistling",
            "kiss",
            "thumping",
            "growling",
            "gunshot",
            "gunshots",
            "applause",
            "buzzer",
            "mumbles",
            "squeaking",
            "popping",
            "gunshots",
            "clicking",
            "claps",
            "silence",
            "silentclapping",
            "clap",
            "laughs",
            "laughing",
            "singing",
            "typing",
            "beeping",
            "groans",
            "unintelligible",
            "bangs",
            "beep",
            "crying",
            "chuckles",
            "swoosh",
            "coughing",
            "indistinct",
            "explosion",
            "blank_audio",
        ]

        # Remove certain words and strings
        for word in removal_list:
            transcribed_text = transcribed_text.replace(word, "")

        # Check if transcription is empty
        if not transcribed_text.strip():
            logger.error(
                "Empty transcription after removal process", extra={"serial_number": connection_id}
            )
            return "silent", 1.2, transcribed_text, "R"

        # Remove whitespaces
        transcribed_text = re.sub(" +", " ", transcribed_text.strip())

        logger.debug(
            f"Processed transcription is: {transcribed_text}",
            extra={"serial_number": connection_id},
        )
        if turn == 2:
            if transcribed_text == "bye":
                transcribed_text = "fine"

        if call_type == "medicare":
            if transcribed_text == "medicare":
                return "N", 1.7, transcribed_text, "F"
        model_used = "SS"
        exact_search = False
        label, confidence, substring_search = self.substring_search(transcribed_text)
        if label == "APM" and call_type == "aca":
            label = "NQA"
        if label == "APA" and call_type == "medicare":
            label = "ABN"
        if label == "APA" or label == "APM":
            label = "AP"
        if label is None:
            model_used = "ES"
            label, confidence, exact_search = self.exact_search(transcribed_text)
            if label == "APM" and call_type == "aca":
                label = "NQA"
            if label == "APA" and call_type == "medicare":
                label = "ABN"
            if label == "APA" or label == "APM":
                label = "AP"

            if label is None:
                # Send the transcribed text to the classification service
                for i in range(3):  # Retry up to 3 times
                    try:
                        data = {
                            "transcribed_text": transcribed_text,
                            "serial_number": connection_id,
                            "model_type": model_type,
                            "call_type": call_type,
                        }
                        response = requests.post(
                            "http://voiceflow_classification:9000/classify/", data=data
                        )
                        response.raise_for_status()
                        response = response.json()
                        label = response.get("label")
                        confidence = response.get("confidence")
                        model_used = response.get("model_used")
                        break
                    except requests.RequestException:
                        time.sleep(0.3 * i)  # Exponential backoff
                    except Exception as e:
                        logger.error(
                            f"Error during classification request: {e}",
                            extra={"serial_number": connection_id},
                        )
                else:
                    model_used = "CE"
                    label = "N"
                    confidence = 1.1
        logger.debug(
            f"label is: {label} and confidence is: {confidence} and exact search is: {exact_search} and "
            f"substring search is: {substring_search}",
            extra={"serial_number": connection_id},
        )
        return label, confidence, transcribed_text, model_used

    def exact_search(self, transcribed_text):
        for search_string, label in self.exact_search_dict.items():
            search_string = search_string.lower()
            if search_string == transcribed_text:
                return label, 1.0, True
        return None, None, False

    def substring_search(self, transcribed_text):
        for search_string, label in self.substring_search_dict.items():
            search_string = search_string.lower()
            if search_string in transcribed_text:
                return label, 1.0, True
        return None, None, False

    @staticmethod
    def check_provider_confirmation(transcribed_text):
        label = None
        classification_time = None
        confidence = 0.0
        # Define the phrases and their possible spelling mistakes
        phrases = [
            "is this xfinity",
            "is this comcast",
            "what company are you",
            "what company is this",
            "are you xfinity",
            "are you comcast",
            "looking for xfinity",
        ]

        alternative_spellings = {
            "xfinity": [
                "xfinity",
                "affinity",
                "afinity",
                "infinity",
                "xfinitycomcast",
                "xfinity comcast",
                "exfinity",
            ],
            "comcast": ["comcast", "com cast", "com"],
        }

        # Create all possible combinations of phrases with the alternative spellings
        alternative_phrases = []
        for phrase in phrases:
            words = phrase.split()
            alternatives = [alternative_spellings.get(word, [word]) for word in words]
            for combination in product(*alternatives):
                alternative_phrases.append(" ".join(combination))

        # Create a pattern for each phrase
        patterns = [
            r"\s*" + r"\s*(?:\w+\W+){0,2}\s*".join(phrase.split()) + r"\s*"
            for phrase in alternative_phrases
        ]

        provider_confirmation_detected = False
        for pattern in patterns:
            match = re.search(pattern, transcribed_text)
            if match:
                label = "provider-confirmation"
                confidence = 1.0
                provider_confirmation_detected = True
                break

        return provider_confirmation_detected, label, confidence, classification_time
