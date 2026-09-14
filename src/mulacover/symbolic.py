"""Shared symbolic conditioning and lossless MIDI interchange for MuLaCover."""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Optional

import torch

MAX_SYMBOLIC_LENGTH = 5000
MIDI_RESOLUTION = 480
_ROOTS = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}
# Exact matches precede the ordered prefix fallback used during model training.
_CHORD_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "5": (0, 7),
}


def _load_mido():
    try:
        import mido
    except ImportError as error:
        raise ImportError(
            "MIDI conditioning requires mido. Install mulacover[midi]."
        ) from error
    return mido


def _empty_notes():
    return torch.empty((0, 3), dtype=torch.long)


def _finite_nonnegative(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite nonnegative number.") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return value


def _validate_bpm(bpm):
    if bpm is None:
        return None
    bpm = _finite_nonnegative(bpm, "bpm")
    if bpm == 0:
        raise ValueError("bpm must be positive.")
    return bpm


def _quantize(value):
    """Round a nonnegative musical position to its nearest sixteenth note."""
    if not math.isfinite(value):
        raise ValueError("Symbolic timing must remain finite after conversion.")
    return math.floor(value + 0.5)


def _validate_triplets(value, name, chord=False):
    if value is None:
        return _empty_notes()
    try:
        notes = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            f"{name} must contain (onset, pitch, duration) triplets."
        ) from error
    if notes.numel() == 0:
        return _empty_notes()
    if notes.ndim != 2 or notes.shape[1] != 3 or notes.is_complex():
        raise ValueError(f"{name} must have shape (number_of_notes, 3).")
    if not torch.isfinite(notes).all() or (notes < 0).any():
        raise ValueError(f"{name} values must be finite and nonnegative.")
    if notes.is_floating_point() and not torch.equal(notes, notes.round()):
        raise ValueError(f"{name} values must be integers on the sixteenth-note grid.")
    if (notes[:, 1] > 127).any():
        raise ValueError(f"{name} MIDI pitches must be between 0 and 127.")
    if chord:
        notes = notes[notes[:, 2] > 0]
    # Check before conversion to int64 or dense allocation, including huge inputs.
    if (notes[:, (0, 2)] > MAX_SYMBOLIC_LENGTH).any() or (
        notes[:, 0] + notes[:, 2] > MAX_SYMBOLIC_LENGTH
    ).any():
        raise ValueError(
            f"Symbolic input exceeds {MAX_SYMBOLIC_LENGTH} sixteenth notes; "
            "trim the input before generating."
        )
    notes = notes.to(torch.long).clone()
    if chord:
        notes[:, 1] %= 12
    else:
        notes[:, 2].clamp_(min=1)
    return notes


def _chord_pitches(label):
    if not isinstance(label, str) or ":" not in label:
        return ()
    root, quality = label.split(":", 1)
    if root not in _ROOTS:
        return ()
    intervals = _CHORD_INTERVALS.get(quality)
    if intervals is None:
        intervals = next(
            (
                notes
                for name, notes in _CHORD_INTERVALS.items()
                if quality.startswith(name)
            ),
            (),
        )
    return tuple((_ROOTS[root] + note) % 12 for note in intervals)


def _read_midi(path, include_drum_channel=False):
    mido = _load_mido()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file does not exist: {path}")
    try:
        midi = mido.MidiFile(path)
    except (OSError, ValueError, EOFError, KeyError) as error:
        raise ValueError(f"Cannot read MIDI file {path}: {error}") from error
    if midi.type not in (0, 1):
        raise ValueError(
            f"MIDI type 2 has independent timelines and is unsupported: {path}"
        )
    if midi.ticks_per_beat <= 0:
        raise ValueError(f"MIDI must use positive PPQ timing, not SMPTE timing: {path}")

    triplets = []
    tempos = []
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        active = defaultdict(deque)
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                tempos.append((tick, message.tempo))
            if message.type not in ("note_on", "note_off"):
                continue
            if not include_drum_channel and message.channel == 9:
                continue
            key = (message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                active[key].append(tick)
                continue
            if not active[key]:
                raise ValueError(
                    f"Unmatched note-off in MIDI track {track_index}: {path}"
                )
            onset = active[key].popleft()
            # Integer arithmetic avoids float error at exact half-grid positions.
            start = (onset * 8 + midi.ticks_per_beat) // (2 * midi.ticks_per_beat)
            end = (tick * 8 + midi.ticks_per_beat) // (2 * midi.ticks_per_beat)
            triplets.append((start, message.note, end - start))
        if any(active.values()):
            raise ValueError(f"Unterminated MIDI note in track {track_index}: {path}")
    # Tempo is metadata: conditioning is always read in beat space, even with
    # tempo changes. Keep the initial tempo for listening to exported MIDI.
    initial_tempo = min(tempos, default=(0, 500000), key=lambda event: event[0])[1]
    if initial_tempo <= 0:
        raise ValueError(f"MIDI tempo must be positive: {path}")
    return triplets, mido.tempo2bpm(initial_tempo)


@dataclass
class SymbolicCondition:
    """Melody, chords and optional drums on a shared sixteenth-note grid.

    Each tensor contains integer ``(onset, pitch, duration)`` rows. Chord
    pitches are normalized modulo 12. BPM describes the source tempo; it is
    not a separate conditioning input and does not fix the generated tempo.
    """

    melody: torch.Tensor
    chords: torch.Tensor
    drums: torch.Tensor = field(default_factory=_empty_notes)
    bpm: Optional[float] = None

    def __post_init__(self):
        self.melody = _validate_triplets(self.melody, "melody")
        self.chords = _validate_triplets(self.chords, "chords", chord=True)
        self.drums = _validate_triplets(self.drums, "drums")
        self.bpm = _validate_bpm(self.bpm)
        if not len(self.melody):
            raise ValueError("Melody must contain at least one note.")
        if self.symbolic_length > MAX_SYMBOLIC_LENGTH:
            raise ValueError(
                f"Symbolic input exceeds {MAX_SYMBOLIC_LENGTH} sixteenth notes; "
                "trim the input before generating."
            )

    @property
    def symbolic_length(self):
        return max(
            (int((notes[:, 0] + notes[:, 2]).max()) if len(notes) else 0)
            for notes in (self.melody, self.drums, self.chords)
        )

    @property
    def length(self):
        return self.symbolic_length

    @classmethod
    def from_midi(cls, melody_path, chord_path, drum_path=None):
        """Merge non-drum melody/chord tracks; an explicit drum file uses all notes."""
        melody, bpm = _read_midi(melody_path)
        chords, _ = _read_midi(chord_path)
        drums = [] if drum_path is None else _read_midi(drum_path, True)[0]
        return cls(melody=melody, chords=chords, drums=drums, bpm=bpm)

    @classmethod
    def from_transcription(cls, notes, chord_events, bpm):
        """Convert second-based notes and ``(label, start_beat, end_beat)`` chords."""
        bpm = _validate_bpm(bpm)
        if bpm is None:
            raise ValueError("Transcribed notes require a positive bpm.")
        melody, chords, drums = [], [], []
        for note in notes:

            def get(name):
                return note[name] if isinstance(note, dict) else getattr(note, name)

            onset = _finite_nonnegative(get("onset"), "note onset")
            offset = _finite_nonnegative(get("offset"), "note offset")
            if offset < onset:
                raise ValueError("Note offset must not precede its onset.")
            start = _quantize(onset * bpm / 15)
            end = _quantize(offset * bpm / 15)
            triplet = (start, get("pitch"), end - start)
            if get("is_drum"):
                drums.append(triplet)
            elif get("program") == 100:
                melody.append(triplet)
        for label, onset, offset in chord_events:
            onset = _finite_nonnegative(onset, "chord onset")
            offset = _finite_nonnegative(offset, "chord offset")
            if offset < onset:
                raise ValueError("Chord offset must not precede its onset.")
            start, end = _quantize(onset * 4), _quantize(offset * 4)
            if end > start:
                chords.extend(
                    (start, pitch, end - start) for pitch in _chord_pitches(label)
                )
        return cls(melody=melody, chords=chords, drums=drums, bpm=bpm)

    def to_tensors(self):
        """Build the onset/sustain, drum and chroma conditioning tensors."""
        self.__post_init__()
        length = self.symbolic_length
        melody = torch.zeros((length, 128, 2), dtype=torch.float32)
        drums = torch.zeros((length, 128), dtype=torch.float32)
        chords = torch.zeros((length, 12), dtype=torch.float32)
        for onset, pitch, duration in self.melody.tolist():
            melody[onset, pitch, 0] = 1
            melody[onset + 1 : onset + duration, pitch, 1] = 1
        for onset, pitch, duration in self.drums.tolist():
            drums[onset : onset + duration, pitch] = 1
        for onset, pitch, duration in self.chords.tolist():
            chords[onset : onset + duration, pitch] = 1
        return {
            "pianoroll": melody.reshape(length, 256),
            "drum_pianoroll": drums,
            "chord": chords,
            "context_mask": torch.zeros(length, dtype=torch.bool),
        }

    def save_midi(self, directory):
        """Export the exact model condition as three replayable MIDI files."""
        self.__post_init__()
        mido = _load_mido()
        directory = Path(directory)
        bpm = self.bpm if self.bpm is not None else 120.0
        tempo = mido.bpm2tempo(bpm)
        if not 0 < tempo < 2**24:
            raise ValueError("bpm cannot be represented by the MIDI tempo field.")
        directory.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name, notes in (
            ("melody", self.melody),
            ("chord", self.chords),
            ("drums", self.drums),
        ):
            midi = mido.MidiFile(type=0, ticks_per_beat=MIDI_RESOLUTION)
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=tempo))
            track.append(mido.MetaMessage("track_name", name=name))
            events = []
            channel = 9 if name == "drums" else 0
            for onset, pitch, duration in notes.tolist():
                pitch = pitch + 60 if name == "chord" else pitch
                events.append((onset * 120, 1, pitch))
                events.append(((onset + duration) * 120, 0, pitch))
            tick = 0
            for event_tick, is_on, pitch in sorted(events):
                track.append(
                    mido.Message(
                        "note_on" if is_on else "note_off",
                        channel=channel,
                        note=pitch,
                        velocity=100 if is_on else 0,
                        time=event_tick - tick,
                    )
                )
                tick = event_tick
            track.append(mido.MetaMessage("end_of_track"))
            paths[name] = directory / f"{name}.mid"
            midi.save(paths[name])
        paths["bpm"] = directory / "bpm.txt"
        paths["bpm"].write_text(f"{bpm:.12g}\n", encoding="utf-8")
        return paths
