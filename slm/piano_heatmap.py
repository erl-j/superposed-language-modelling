import numpy as np
import matplotlib.pyplot as plt

def create_piano_visualization(probabilities, title="Piano Visualization"):
    """
    Plot probabilities for MIDI pitches 21-104 (A0 to C8) as a bar plot.
    Args:
        probabilities: shape (7,12) or (84,) for MIDI 21-104
        title: plot title
    """
    # Flatten if needed
    probs = probabilities.flatten()
    assert probs.shape[0] == 84, "Input must cover MIDI 21-104 (84 notes)"
    midi_pitches = np.arange(21, 105)

    # Label only C notes
    xticks = []
    xlabels = []
    for i, midi in enumerate(midi_pitches):
        if midi % 12 == 0:  # C notes
            xticks.append(i)
            octave = (midi // 12) - 1
            xlabels.append(f"C{octave}")

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.bar(np.arange(84), probs, width=0.8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_xlabel("MIDI Note (C notes labeled)")
    ax.set_ylabel("Probability")
    ax.set_title(title)
    # Add vertical lines at each C note
    for tick in xticks:
        ax.axvline(x=tick, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    return fig

# Example usage
if __name__ == "__main__":
    # Create some example data (higher probabilities in the middle octaves)
    data = np.zeros((7, 12))
    data[3:5, :] = 0.8  # Middle octaves have high probability
    data[2:6, 0:2] = 0.5  # Some C and C# notes in surrounding octaves

    # Create and show the plot
    fig = create_piano_visualization(data, "Example Piano Visualization")
    plt.show() 