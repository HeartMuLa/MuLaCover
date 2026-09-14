"""Inference Viterbi decoder adapted from Music X Lab (MIT; see LICENSE)."""

from pathlib import Path

import numpy as np

from .complex_chord import Chord, NUM_TO_ABS_SCALE, shift_complex_chord_array


class ChordDecoder:
    def __init__(self, transition_penalty: float = 30.0):
        self.transition_penalty = transition_penalty
        template_path = Path(__file__).with_name("chord_vocabulary.txt")
        chord_pool = {}
        with template_path.open(encoding="utf-8") as file:
            for line in file:
                name = line.strip()
                if ":" not in name:
                    continue
                root, suffix = name.split(":")
                if root != "C":
                    raise ValueError(f"Unexpected chord template root: {name}")
                chord = Chord(name).to_numpy()
                if -2 in chord:
                    continue
                for shift in range(12):
                    shifted = tuple(shift_complex_chord_array(chord, shift))
                    if shifted not in chord_pool:
                        chord_pool[shifted] = f"{NUM_TO_ABS_SCALE[shift]}:{suffix}"
        self.chords = [((0, -1, -1, -1, -1, -1), "N")]
        self.chords.extend(chord_pool.items())

    def observations(self, probabilities):
        names, chords = [], []
        for chord, name in self.chords:
            if any(chord[i] >= probabilities[i].shape[-1] for i in range(6)):
                continue
            names.append(name)
            chords.append(chord)
        chords = np.asarray(chords, dtype=np.int64)
        # The bass head reserves index zero for no chord.
        chords[:, 1] += 1
        with np.errstate(divide="ignore"):
            observations = np.log(probabilities[0][:, chords[:, 0]])
            observations += np.log(probabilities[1][:, chords[:, 1]])
            for suffix in range(4):
                selected = chords[:, suffix + 2] >= 0
                observations[:, selected] += np.log(
                    probabilities[suffix + 2][:, chords[selected, suffix + 2]]
                )
        return names, observations

    def decode(self, probabilities):
        """Decode unrestricted frame transitions, as in chord_recognition.py."""
        names, observations = self.observations(probabilities)
        frame_count, chord_count = observations.shape
        if frame_count == 0:
            return []
        scores = np.zeros_like(observations)
        scores[0, 1:] = -np.inf
        scores[0] += observations[0]
        best = np.zeros(frame_count, dtype=np.int64)
        previous = np.zeros((frame_count, chord_count), dtype=np.int64)
        best[0] = np.argmax(scores[0])
        previous[0] = -1
        indices = np.arange(chord_count)
        for frame in range(1, frame_count):
            same = scores[frame - 1]
            different = scores[frame - 1, best[frame - 1]] - self.transition_penalty
            keep = same > different
            scores[frame] = np.maximum(different, same) + observations[frame]
            previous[frame] = best[frame - 1]
            previous[frame, keep] = indices[keep]
            best[frame] = np.argmax(scores[frame])
        decoded = np.zeros(frame_count, dtype=np.int64)
        decoded[-1] = best[-1]
        for frame in range(frame_count - 2, -1, -1):
            decoded[frame] = previous[frame + 1, decoded[frame + 1]]
        return [names[index] for index in decoded]

    def decode_segments(self, probabilities, frame_seconds: float):
        labels = self.decode(probabilities)
        segments = []
        start = 0
        for frame, label in enumerate(labels):
            if frame + 1 == len(labels) or labels[frame + 1] != label:
                segments.append(
                    (label, start * frame_seconds, (frame + 1) * frame_seconds)
                )
                start = frame + 1
        return segments
