import asyncio
import collections
import time

import torch
from faster_whisper import WhisperModel

from voiceflow_ai.core.config import settings as c
from voiceflow_ai.core.logger import get_logger

logger = get_logger("TranscriptionService")


class TranscriptionService:
    def __init__(self):
        self.whisper_model = None
        self.shutdown_in_progress = False
        self.active_transcriptions_count = 0
        self.active_transcriptions = collections.deque()
        self.test_file = c.MODEL_LOAD_FILEPATH

    def initialize_model(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_size = "small.en"
            logger.info(f"device is: {device}")
            if device == "cpu":
                self.whisper_model = WhisperModel(model_size, device=device, compute_type="int8")
            else:
                self.whisper_model = WhisperModel(
                    model_size, device=device, compute_type="float16", num_workers=1
                )
            logger.info(f"Whisper loaded and model size is: {model_size}")
            time.sleep(1)
            self.transcribe_audio(self.test_file, "medicare", "A", 1)
            logger.info("Whisper model has been loaded to the device")
            return True
        except Exception as e:
            logger.error(f"Error during whisper initialization: {e}", exc_info=True)
            return False

    def transcribe_audio(self, file_path, call_type, model_type, turn_number=1):
        try:
            transcribed_text = None
            self.active_transcriptions_count += 1
            self.active_transcriptions.append(time.time())
            initial_prompt = None
            if self.whisper_model is not None and c.TYPE:
                if turn_number == 4 and call_type == "medicare":
                    initial_prompt = (
                        "fifty-one, fifty-two, fifty-three, fifty-four, fifty-five, fifty-six, "
                        "fifty-seven, fifty-eight, fifty-nine, sixty, sixty-one, sixty-two, sixty-three, "
                        "sixty-four, sixty-five, sixty-six, sixty-seven, sixty-eight, sixty-nine, "
                        "seventy, seventy-one, seventy-two, seventy-three, seventy-four, seventy-five, "
                        "seventy-six, seventy-seven, seventy-eight, seventy-nine, eighty, eighty-one, "
                        "eighty-two, eighty-three, eighty-four, eighty-five, eighty-six, eighty-seven, "
                        "eighty-eight, eighty-nine, ninety, ninety-one, ninety-two, ninety-three, "
                        "ninety-four, ninety-five, ninety-six, ninety-seven, ninety-eight, ninety-nine, "
                        "one hundred, one hundred and one, one hundred and two, one hundred and three, "
                        "one hundred and four, one hundred and five, one hundred and six, "
                        "one hundred and seven, one hundred and eight, one hundred and nine, "
                        "one hundred and ten"
                    )

                elif turn_number == 6 and call_type == "aca" and model_type == "A":
                    initial_prompt = (
                        "Yes, no, I do, I don't, yes I do, no I don't, sure, I'm sure, yeah, nah, absolutely, of course, "
                        "true, false, it is, it's, more, more than, less, less than, none, sir, ma'am, higher, lower, "
                        "employed, unemployed, making, nothing, I make more, I make less, how, what, who, how much, say, "
                        "I don't know, interested, call, list, hang up, recording, recorded, transfer, invalid, business, "
                        "10000, 12000, 13000, 15000, 20000, 25000, 30000, 35000, 40000, 43000, 50000, 55000, 60000, 80000, "
                        "90000, 100000, 120000, 150000, 200000, 1000, 1500, 2000, 3000, 150, 200, 350, dollars, a year, a month, "
                        "a week, an hour, income"
                    )
                elif turn_number == 5 and call_type == "aca" and model_type == "A":
                    initial_prompt = (
                        "fifty-one, fifty-two, fifty-three, fifty-four, fifty-five, fifty-six, "
                        "fifty-seven, fifty-eight, fifty-nine, sixty, sixty-one, sixty-two, sixty-three, "
                        "sixty-four, sixty-five, sixty-six, sixty-seven, sixty-eight, sixty-nine, "
                        "seventy, seventy-one, seventy-two, seventy-three, seventy-four, seventy-five,"
                        "seventy-six, seventy-seven, seventy-eight, seventy-nine, eighty, eighty-one, "
                        "eighty-two, eighty-three, eighty-four, eighty-five, eighty-six, eighty-seven, "
                        "eighty-eight, eighty-nine, ninety, ninety-one, "
                        "yes, no, I am, years, old, under, over, age, born, "
                        "twenty, twenty-four, twenty-nine, thirty, thirty-five, thirty-nine,"
                        "forty, forty-seven, forty-one, nineteen, forty-two"
                    )

                elif turn_number == 4 and call_type == "aca" and model_type == "B":
                    initial_prompt = (
                        "fifty-one, fifty-two, fifty-three, fifty-four, fifty-five, fifty-six, "
                        "fifty-seven, fifty-eight, fifty-nine, sixty, sixty-one, sixty-two, sixty-three, "
                        "sixty-four, sixty-five, sixty-six, sixty-seven, sixty-eight, sixty-nine, "
                        "seventy, seventy-one, seventy-two, seventy-three, seventy-four, seventy-five,"
                        "seventy-six, seventy-seven, seventy-eight, seventy-nine, eighty, eighty-one, "
                        "eighty-two, eighty-three, eighty-four, eighty-five, eighty-six, eighty-seven, "
                        "eighty-eight, eighty-nine, ninety, ninety-one, "
                        "yes, no, I am, years, old, under, over, age, born, "
                        "twenty, twenty-four, twenty-nine, thirty, thirty-five, thirty-nine,"
                        "forty, forty-seven, forty-one, nineteen, forty-two"
                    )

                elif turn_number == 3 and call_type == "fe":
                    initial_prompt = (
                        "fifty-one, fifty-two, fifty-three, fifty-four, fifty-five, fifty-six, "
                        "fifty-seven, fifty-eight, fifty-nine, sixty, sixty-one, sixty-two, sixty-three, "
                        "sixty-four, sixty-five, sixty-six, sixty-seven, sixty-eight, sixty-nine, "
                        "seventy, seventy-one, seventy-two, seventy-three, seventy-four, seventy-five,"
                        "seventy-six, seventy-seven, seventy-eight, seventy-nine, eighty, eighty-one, "
                        "eighty-two, eighty-three, eighty-four, eighty-five, eighty-six, eighty-seven, "
                        "eighty-eight, eighty-nine, ninety, ninety-one, "
                        "yes, no, I am, years, old, under, over, age, born, "
                        "twenty, twenty-four, twenty-nine, thirty, thirty-five, thirty-nine,"
                        "forty, forty-seven, forty-one, nineteen, forty-two"
                    )

                elif turn_number == 2 and call_type == "aca":
                    initial_prompt = (
                        "calling, want, need, what, hello, yes, no, sure, nah, not,"
                        "connect,for, you, fine, how about you, I'm fine,"
                        "yes sir, I'm, doing, alright, call, cannot, about, where, help, too, doing good, this, "
                        "not, good, how, english, spanish, speak, I, may, hi, "
                        "busy, moment, right, calling, help, pretty, yup, you, at, work, right now, "
                        "I don't, I do, who, who's, quit, do, don't, what, and, just fine, and you, "
                        "yeah, go ahead, help, going, can, I'm doing fine, I'm good, about"
                    )
                elif turn_number == 2 and call_type == "fe":
                    initial_prompt = (
                        "calling, want, need, what, hello, yes, no, sure, nah, not,"
                        "connect,for, you, fine, how about you, I'm fine,"
                        "yes sir, I'm, doing, alright, call, cannot, about, where, help, too, doing good, this, "
                        "not, good, how, english, spanish, speak, I, may, hi, "
                        "busy, moment, right, calling, help, pretty, yup, you, at, work, right now, "
                        "I don't, I do, who, who's, quit, do, don't, what, and, just fine, and you, "
                        "yeah, go ahead, help, going, can, I'm doing fine, I'm good, about"
                    )
                elif turn_number == 2 and call_type == "medicare" and model_type == "B":
                    initial_prompt = (
                        "calling, want, need, what, hello, yes, no, sure, nah, not,"
                        "connect,for, you, fine, how about you, I'm fine,"
                        "yes sir, I'm, doing, alright, call, cannot, about, where, help, too, doing good, this, "
                        "not, good, how, english, spanish, speak, I, may, hi, "
                        "busy, moment, right, calling, help, pretty, yup, you, at, work, right now, "
                        "I don't, I do, who, who's, quit, do, don't, what, and, just fine, and you, "
                        "yeah, go ahead, help, going, can, I'm doing fine, I'm good, about"
                    )
                elif turn_number == 4 and call_type == "fe":
                    initial_prompt = (
                        "yes, yeah, no, I do, I don't, absolutely, absolutely not, no I don't, sure, correct, "
                        "incorrect, right, not, have, I have, do, don't, do what, interested, not interested, busy, "
                        "stop, quit, take, take me, calling, call, remove, list, calling list, please, again, have "
                        "what, any what, need, want, employed, employment, coverage, covered, old, retired, subsidy, "
                        "qualified, eligible, not eligible, guess, I guess, through, is through, my insurance, "
                        "entitlement, decisions, decide, final expense, burial, funeral, military, veteran, a veteran, "
                        "social security, plan, card, work, back, say, say what, what, what now, why, who, "
                        "Do I have, alright, nah, yes sir, no sir, english, spanish"
                    )

                elif (
                    (turn_number == 3 or turn_number == 4)
                    and call_type == "aca"
                    and model_type == "A"
                ):
                    initial_prompt = (
                        "yes, yeah, no, I do, I don't, absolutely, absolutely not, no I don't, sure, correct, incorrect, right, not, "
                        "have, I have, do, don't, do what, interested, not interested, busy, stop, quit, take, take me, calling, call, "
                        "remove, list, calling list, please, again, have what, any what, need, want, employed, employment, coverage, covered, "
                        "old, retired, subsidy, qualified, eligible, not eligible, guess, I guess, through, is through, my insurance, entitlement, "
                        "not entitled, citizen, healthcare, insurance, insured, ACA, affordable care act, blue cross, blue shield, ambetter, "
                        "aetna, humana, oscar, united, united health, Medicare, have medicare, Medicaid, have medicaid, tricare, VA, the VA, "
                        "military, veteran, a veteran, social security, plan, card, work, back, say, say what, what, what now, why, who, "
                        "Do I have, alright, nah, yes sir, no sir, english, spanish"
                    )

                elif (
                    (turn_number == 3 or turn_number == 5)
                    and call_type == "aca"
                    and model_type == "B"
                ):
                    initial_prompt = (
                        "yes, yeah, no, I do, I don't, absolutely, absolutely not, no I don't, sure, correct, incorrect, right, not, "
                        "have, I have, do, don't, do what, interested, not interested, busy, stop, quit, take, take me, calling, call, "
                        "remove, list, calling list, please, again, have what, any what, need, want, employed, employment, coverage, covered, "
                        "old, retired, subsidy, qualified, eligible, not eligible, guess, I guess, through, is through, my insurance, entitlement, "
                        "not entitled, citizen, healthcare, insurance, insured, ACA, affordable care act, blue cross, blue shield, ambetter, "
                        "aetna, humana, oscar, united, united health, Medicare, have medicare, Medicaid, have medicaid, tricare, VA, the VA, "
                        "military, veteran, a veteran, social security, plan, card, work, back, say, say what, what, what now, why, who, "
                        "Do I have, alright, nah, yes sir, no sir, english, spanish"
                    )

                elif turn_number == 1:
                    if call_type == "medicare":
                        initial_prompt = (
                            "what, okay, go ahead, speak, do you, need, hello, yes, yeah, call, who's calling, hi, "
                            "hey, this, want, why, calling, leave, message, not available, sorry,thank you, bye"
                        )
                    if call_type == "aca":
                        initial_prompt = (
                            "what, okay, go ahead, speak, do you, need, hello, yes, yeah, call, who's calling, hi, "
                            "hey, this, want, why, calling, leave, message, not available, sorry"
                        )
                    if call_type == "fe":
                        initial_prompt = (
                            "what, okay, go ahead, speak, do you, need, hello, yes, yeah, call, who's calling, hi, "
                            "hey, this, want, why, calling, leave, message, not available, sorry"
                        )
                elif call_type == "medicare":
                    initial_prompt = (
                        "calling, want, need, what, tricare, minute, hello, yes, no, sure, nah, correct, "
                        "connect, incorrect, military, both, interested, not interested, retired, "
                        "medicare, Medicaid, humana, above, over, yes sir, senior alright, part A, card, "
                        "part B, part C, A, B, C, D, V-A, A and B, and C, ABC and D, AB and C, E, not, "
                        "entitled, busy, moment, eyes, hearing, dental, advantage, not entitled, right, "
                        "absolutely, need, mate, united, invited, whole, social security, I don't, I do, "
                        "call, vision, medication, yeah, guess, cover, I guess, go ahead"
                    )
                elif call_type == "aca":
                    initial_prompt = (
                        "calling, want, need, what, tricare, minute, hello, yes, no, sure, nah, correct, "
                        "connect, incorrect, military, both, interested, not interested, retired, insurance, "
                        "medicare, employment, Medicaid, humana, above, over, yes sir, senior, I am, alright, card, "
                        "ACA, aetna, humana, united, M-better, less, more, annual, income, not, "
                        "entitled, busy, moment, eyes, hearing, dental, advantage, not entitled, right, "
                        "absolutely, need, mate, united, invited, whole, social security, I don't, I do, "
                        "call, vision, medication, yeah, guess, cover, I guess, go ahead, V-A, employed"
                    )
                else:
                    initial_prompt = None

                logger.info(f"received file {file_path}")
                with open(file_path, "rb") as f:
                    segments, _ = self.whisper_model.transcribe(
                        f,
                        beam_size=5,
                        best_of=5,
                        initial_prompt=initial_prompt,
                        suppress_tokens=[0, 11, 13, 30],
                    )
                    transcribed_text = " ".join([segment.text for segment in segments])

            elif self.whisper_model is not None and not c.TYPE:
                with open(file_path, "rb") as f:
                    segments, _ = self.whisper_model.transcribe(
                        f,
                        beam_size=5,
                        best_of=5,
                        initial_prompt="Glossary: new service, customer support, prices, packages, subscription, "
                        "phone, cable, internet, help, TV, mobile, bill, router, wires, hello, hi, "
                        "sign up, signing up, billed, customer, cost, wi-fi, wifi, apartment, xfinity, "
                        "comcast, AT&T, optimum, spectrum, viacast, verizon, t-mobile, "
                        "cox communication, address, moving, deals, transfer, house, line, agent, "
                        "hang up, hung up, dropped, hanged, screen, program, cancelled, pricing, home, "
                        "restart, area, set up, hooked, old, account, looking, questions, calling, "
                        "transferred, hook up, different, quote, how much, information, contract, "
                        "contact, turned off, on, disabled, qualify, the service, move the service, "
                        "support, Ima, suspend, disconnect, disconnected",
                        suppress_tokens=[0, 11, 13, 30],
                    )
                    transcribed_text = " ".join([segment.text for segment in segments])

            self.active_transcriptions_count -= 1
            self.active_transcriptions.popleft()
            return transcribed_text

        except Exception as error:
            logger.error(f"Error during transcription: {error}")
            self.active_transcriptions -= 1
        raise

    async def shutdown(self):
        self.shutdown_in_progress = True
        while self.active_transcriptions_count > 0:
            await asyncio.sleep(0.1)
        self.whisper_model = None
        logger.info("Graceful shutdown completed.")
