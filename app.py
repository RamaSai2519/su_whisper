import os
import wave
import torch
import whisper
import datetime
import contextlib
import numpy as np
from pydub import AudioSegment
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


EMBED_MODEL_PATH = os.getcwd() + '/models'


class AudioProcessor:
    def __init__(self, path: str):
        self.path = path
        self.audio_segment = AudioSegment.from_mp3(path)
        self.wav_path = path.replace('.mp3', '.wav')
        self.audio_segment.export(self.wav_path, format="wav")
        self.duration = self.get_duration()

    def get_duration(self) -> float:
        with contextlib.closing(wave.open(self.wav_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            return frames / float(rate)


class Transcriber:
    def __init__(self, model_name: str = "turbo"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, path: str) -> list:
        result = self.model.transcribe(path)
        return result["segments"]


class SpeakerIdentifier:
    def __init__(self, segments: list, duration: float, path: str):
        self.segments = segments
        self.duration = duration
        self.path = path
        self.audio = Audio()
        self.embedding_model = EncoderClassifier.from_hparams(
            source=os.path.abspath(EMBED_MODEL_PATH),
            run_opts={"device": "cpu"}
        )
        self.embeddings = self.get_embeddings()
        self.labels = self.cluster_embeddings()

    def segment_embedding(self, segment: dict) -> np.ndarray:
        start = segment["start"]
        end = min(self.duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = self.audio.crop(self.path, clip)

        # waveform shape: [1, time] is expected
        waveform = waveform.squeeze(0).unsqueeze(0)  # ensure [1, time]

        embedding = self.embedding_model.encode_batch(waveform)
        return embedding.squeeze().detach().numpy()

    def get_embeddings(self) -> np.ndarray:
        embeddings = np.zeros(shape=(len(self.segments), 192))
        for i, segment in enumerate(self.segments):
            embeddings[i] = self.segment_embedding(segment)
        return np.nan_to_num(embeddings)

    def cluster_embeddings(self) -> np.ndarray:
        clustering = AgglomerativeClustering().fit(self.embeddings)
        return clustering.labels_

    def assign_speakers(self) -> list:
        for i in range(len(self.segments)):
            self.segments[i]["speaker"] = 'SPEAKER ' + str(self.labels[i] + 1)
        return self.segments


class TranscriptWriter:
    @staticmethod
    def time(secs: int) -> datetime.timedelta:
        return datetime.timedelta(seconds=round(secs))

    def write_transcript(self, segments: list, file_path: str = "transcript.txt"):
        with open(file_path, "w") as f:
            for i, segment in enumerate(segments):
                if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                    f.write("\n" + segment["speaker"] + ' ' +
                            str(self.time(segment["start"])) + '\n')
                f.write(segment["text"][1:] + ' ')


def main() -> None:
    path = 'test.mp3'
    audio_processor = AudioProcessor(path)
    transcriber = Transcriber()
    segments = transcriber.transcribe(path)
    speaker_identifier = SpeakerIdentifier(
        segments, audio_processor.duration, path)
    segments = speaker_identifier.assign_speakers()
    transcript_writer = TranscriptWriter()
    transcript_writer.write_transcript(segments)


if __name__ == "__main__":
    main()
