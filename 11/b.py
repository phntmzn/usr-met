import numpy as np

# MIDI note values for each note
notes = {
    'C': 60, 'C#': 61, 'D': 62, 'D#': 63, 'E': 64, 'F': 65, 'F#': 66, 'G': 67, 
    'G#': 68, 'A': 69, 'A#': 70, 'B': 71
}

# Scales dictionary with MIDI note values
scales = {
    "C MAJOR": [60, 62, 64, 65, 67, 69, 71, 72], 
    "C# MAJOR": [61, 63, 65, 66, 68, 70, 72, 73], 
    "D MAJOR": [62, 64, 66, 67, 69, 71, 73, 74],
    "D# MAJOR": [63, 65, 67, 68, 70, 72, 74, 75],
    "E MAJOR": [64, 66, 68, 69, 71, 73, 75, 76],
    "F MAJOR": [65, 67, 69, 70, 72, 74, 76, 77],
    "F# MAJOR": [66, 68, 70, 71, 73, 75, 77, 78],
    "G MAJOR": [67, 69, 71, 72, 74, 76, 78, 79],
    "G# MAJOR": [68, 70, 72, 73, 75, 77, 79, 80],
    "A MAJOR": [69, 71, 73, 74, 76, 78, 80, 81],
    "A# MAJOR": [70, 72, 74, 75, 77, 79, 81, 82],
    "B MAJOR": [71, 73, 75, 76, 78, 80, 82, 83],

    "C MINOR": [60, 62, 63, 65, 67, 68, 70, 72],
    "C# MINOR": [61, 63, 64, 66, 68, 69, 71, 73],
    "D MINOR": [62, 64, 65, 67, 69, 70, 72, 74],
    "D# MINOR": [63, 65, 66, 68, 70, 71, 73, 75],
    "E MINOR": [64, 66, 67, 69, 71, 72, 74, 76],
    "F MINOR": [65, 67, 68, 69, 71, 72, 74, 75],
    "F# MINOR": [66, 68, 69, 71, 73, 74, 76, 78],
    "G MINOR": [67, 69, 70, 72, 74, 75, 77, 79],
    "G# MINOR": [68, 70, 71, 73, 75, 76, 78, 80],
    "A MINOR": [69, 71, 72, 74, 76, 77, 79, 81],
    "A# MINOR": [70, 72, 73, 75, 77, 78, 80, 82],
    "B MINOR": [71, 73, 74, 76, 78, 79, 81, 83]
}

# Time value durations for different note lengths
time_value_durations = {
    "whole_note": 4,
    "dotted_whole_note": 6,
    "half_note": 2,
    "dotted_half_note": 3,
    "quarter_note": 1,
    "dotted_quarter_note": 1.5,
    "eighth_note": 0.5,
    "dotted_eighth_note": 0.75,
    "sixteenth_note": 0.25,
    "dotted_sixteenth_note": 0.375,
    "thirty_second_note": 0.125,
    "dotted_thirty_second_note": 0.1875,
    "sixty_fourth_note": 0.0625,
    "dotted_sixty_fourth_note": 0.09375,
    "one_hundred_twenty_eighth_note": 0.03125,
    "dotted_one_hundred_twenty_eighth_note": 0.046875,
    "two_hundred_fifty_sixth_note": 0.015625,
    "dotted_two_hundred_fifty_sixth_note": 0.0234375
}

# Chords dictionary with triads, 7th, extended, suspended, added note, and altered chords
chords = {
    'Major': np.array([0, 4, 7]),
    'Minor': np.array([0, 3, 7]),
    'Diminished': np.array([0, 3, 6]),
    'Augmented': np.array([0, 4, 8]),
    'Major 7th': np.array([0, 4, 7, 11]),
    'Minor 7th': np.array([0, 3, 7, 10]),
    'Dominant 7th': np.array([0, 4, 7, 10]),
    'Diminished 7th': np.array([0, 3, 6, 9]),
    'Half-Diminished 7th': np.array([0, 3, 6, 10]),
    'Augmented 7th': np.array([0, 4, 8, 10]),
    '9th': np.array([0, 4, 7, 10, 14]),
    'Major 9th': np.array([0, 4, 7, 11, 14]),
    'Minor 9th': np.array([0, 3, 7, 10, 14]),
    '11th': np.array([0, 4, 7, 10, 14, 17]),
    '13th': np.array([0, 4, 7, 10, 14, 21]),
    'Suspended 2nd (sus2)': np.array([0, 2, 7]),
    'Suspended 4th (sus4)': np.array([0, 5, 7]),
    '6th': np.array([0, 4, 7, 9]),
    'Minor 6th': np.array([0, 3, 7, 9]),
    'Add 9': np.array([0, 4, 7, 14]),
    'Minor Add 9': np.array([0, 3, 7, 14]),
    '7th Flat 5': np.array([0, 4, 6, 10]),
    '7th Sharp 5': np.array([0, 4, 8, 10]),
    'Major 7th Sharp 5': np.array([0, 4, 8, 11]),
    'Minor 7th Flat 5': np.array([0, 3, 6, 10])
}

# Inversions of various chords
inversions = {
    "Major": {
        "Inversion 0": np.array([0, 4, 7]),
        "Inversion 1": np.array([4, 7, 12]),
        "Inversion 2": np.array([7, 12, 16])
    },
    "Minor": {
        "Inversion 0": np.array([0, 3, 7]),
        "Inversion 1": np.array([3, 7, 12]),
        "Inversion 2": np.array([7, 12, 15])
    },
    "Diminished": {
        "Inversion 0": np.array([0, 3, 6]),
        "Inversion 1": np.array([3, 6, 12]),
        "Inversion 2": np.array([6, 12, 15])
    },
    "Augmented": {
        "Inversion 0": np.array([0, 4, 8]),
        "Inversion 1": np.array([4, 8, 12]),
        "Inversion 2": np.array([8, 12, 16])
    },
    "Major 7th": {
        "Inversion 0": np.array([0, 4, 7, 11]),
        "Inversion 1": np.array([4, 7, 11, 12]),
        "Inversion 2": np.array([7, 11, 12, 16]),
        "Inversion 3": np.array([11, 12, 16, 19])
    },
    "Minor 7th": {
        "Inversion 0": np.array([0, 3, 7, 10]),
        "Inversion 1": np.array([3, 7, 10, 12]),
        "Inversion 2": np.array([7, 10, 12, 15]),
        "Inversion 3": np.array([10, 12, 15, 19])
    },
    "9 sus": {
        "Inversion 0": np.array([0, 2, 7, 10]),
        "Inversion 1": np.array([2, 7, 10, 12]),
        "Inversion 2": np.array([7, 10, 12, 14]),
        "Inversion 3": np.array([10, 12, 14, 19])
    },
    "6": {
        "Major 6": {
            "Inversion 0": np.array([0, 4, 7, 9]),
            "Inversion 1": np.array([4, 7, 9, 12]),
            "Inversion 2": np.array([7, 9, 12, 16]),
            "Inversion 3": np.array([9, 12, 16, 19])
        },
        "Minor 6": {
            "Inversion 0": np.array([0, 3, 7, 9]),
            "Inversion 1": np.array([3, 7, 9, 12]),
            "Inversion 2": np.array([7, 9, 12, 15]),
            "Inversion 3": np.array([9, 12, 15, 19])
        }
    },
    "5": {
        "Inversion 0": np.array([0, 7]),
        "Inversion 1": np.array([7, 12]),
        "Inversion 2": np.array([12, 19])
    }
}

# Modes based on interval patterns relative to tonic (in semitones)
modes = {
    "Ionian": np.array([0, 2, 4, 5, 7, 9, 11]),       # Major scale
    "Dorian": np.array([0, 2, 3, 5, 7, 9, 10]),
    "Phrygian": np.array([0, 1, 3, 5, 7, 8, 10]),
    "Lydian": np.array([0, 2, 4, 6, 7, 9, 11]),
    "Mixolydian": np.array([0, 2, 4, 5, 7, 9, 10]),
    "Aeolian": np.array([0, 2, 3, 5, 7, 8, 10]),      # Natural minor
    "Locrian": np.array([0, 1, 3, 5, 6, 8, 10])
}