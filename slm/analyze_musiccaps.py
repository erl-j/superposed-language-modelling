#%%
import pandas as pd


src_dir = "data/musiccaps-public.csv"

# ls current dir

df = pd.read_csv(src_dir)

# Convert aspect_list column to list of strings and flatten
all_aspects = []
for aspects in df['aspect_list']:
    # Convert string representation of list to actual list and extend
    try:
        aspect_list = eval(aspects)
        all_aspects.extend(aspect_list)
    except:
        continue


# Count frequency of each aspect
from collections import Counter
aspect_counts = Counter(all_aspects)

# Convert to pandas DataFrame and sort by count
aspect_df = pd.DataFrame.from_dict(aspect_counts, orient='index', columns=['count'])
aspect_df = aspect_df.sort_values('count', ascending=False)
aspect_df.index.name = 'aspect'
aspect_df = aspect_df.reset_index()

print("\nAspect counts:")
print(aspect_df)

# write to file
aspect_df.to_csv('aspect_counts.csv', index=False)

# print plot of distribution (zipf style)
import matplotlib.pyplot as plt

# Create a line plot of aspect counts
plt.figure(figsize=(12, 6))

plt.plot(range(len(aspect_df)), aspect_df['count'])
plt.xlabel('Aspect Rank')
plt.ylabel('Count')
plt.title('Distribution of Aspect Counts')
plt.grid(True)
plt.show()

# Count tokens by splitting on spaces and punctuation
import re

# Combine all captions into one string
all_captions = ' '.join(df['caption'].fillna(''))

# Split on spaces and punctuation, keeping only non-empty tokens, - and /
tokens = [token for token in re.split(r'[^\w\-/]+', all_captions.lower()) if token]

# Count frequency of each token
token_counts = Counter(tokens)

# Convert to DataFrame and sort
token_df = pd.DataFrame.from_dict(token_counts, orient='index', columns=['count'])
token_df = token_df.sort_values('count', ascending=False)
token_df.index.name = 'token'
token_df = token_df.reset_index()

print("\nToken counts:")
print(token_df)

# Write to file
token_df.to_csv('token_counts.csv', index=False)

# Plot distribution
plt.figure(figsize=(12, 6))
plt.plot(range(len(token_df)), token_df['count'])
plt.xlabel('Token Rank')
plt.ylabel('Count')
plt.title('Distribution of Token Counts')
plt.grid(True)
plt.show()






#%%



# high precision, low recall.

# midi_prompts = [
#     {
#         "generate_prompt": "F# minor trap progression, 140 BPM. Half-time feel. Use octave jumps in bass. Add melodic tension with tritones.",
#         # F#minor, programatic. Pitch class histogram. Check root.
#         # 140 BPM, check/enforce.
#         # Half-time feel. Snare has to happen on three. Compare density and snare and kick.
#         # Use octave jumps in bass. Check if two following notes have an octave apart.
#         # Is tritone interval used in.
#         "edit_prompt": "Take this loop and transpose down minor 3rd. Remove every third bass note. Double melody an octave up."
#         # this is a tool use thing.
#         # check output.
#         # No need to check for musicality of output.
#     },
#     {
#         "generate_prompt": "Jazzy D Dorian chord sequence, 85 BPM. 7th and 9th extensions. Walking bass. Subtle swing.",
#         # Jazzy, use clap?
#         # Chord sequence. Polyphony. More than 1 onset group.
#         # 85 bpm. Check/enforce.
#         # 7th and 9th extensions. Check that there are 7 from the root.
#         # Walking bass, No polyphony. Regular interonset time. # CLAP?
#         # Subtle swing, where does 8th notes and 16 notes land?
#         "edit_prompt": "Take this loop and convert 4/4 to 3-3-2 pattern. Shift accent to offbeats. Modulate to relative minor at bar 9."
#         # 3-3-2 patterns. Count onsets on 3-3-2. HARD. Procedural?
#         # Velocity curve. Check that accents are on offbeats.
#         # Modulate to relative minor at bar 9. -3 semitones, major to minor.
#     },
#     {
#         "generate_prompt": "C# minor future garage, 165 BPM. Syncopated rhythm. Create polyrhythmic variation between hi-hats and kick.",
#         # C# minor, programatic
#         # Syncopated rythm. 8th notes and 16th notes. CHECK MIR LIBRARY
#         # Polyrythmic variation between hihat and kicks. 
#         # Best fitting grid.
#         # Check periodicity of kick and snare.
#         "edit_prompt": "Take this loop and reverse melody notes. Create tension with +1 semitone rise every 4 bars. Drop octave for final section."
    
#     },
#     {
#         "generate_prompt": "B Phrygian dominant arpeggio, 118 BPM. 16th note pattern with accent on offbeats. Four-bar phrase with tension-release.",
#         "edit_prompt": "Take this loop and halve tempo. Add triplet ghost notes. Replace every 4th chord with relative minor. Create rhythmic variation."
#     },
#     {
#         "generate_prompt": "128 BPM techno sequence. 16-step bassline pattern with rests on 7 and 15. Evolving kick pattern.",
#         "edit_prompt": "Take this loop and rearrange into 8-bar harmonic movement. Keep same notes but shuffle rhythm. Add chromatic passing tones."
#     },
#     {
#         "generate_prompt": "Orchestral tension builder, 90 BPM. Rising minor scale runs. 5:4 polyrhythm. Build to climax at bar 16.",
#         "edit_prompt": "Take this loop and convert to half-time feel. Create call-response by muting strategic sections. Mirror melody in second half."
#     },
#     {
#         "generate_prompt": "Neo-soul progression, 96 BPM. Add suspended notes and voice leading. Push rhythm slightly behind grid.",
#         "edit_prompt": "Take this loop and arpeggiate all chords. Switch between ascending and descending patterns. Add octave jumps every 4th note."
#     },
#     {
#         "generate_prompt": "Lydian chord progression, 150 BPM. Build with 4-bar phrases. Drop to half-time on bar 17. Tension on 7th scale degree.",
#         "edit_prompt": "Take this loop and chop into 8th notes. Rearrange in Fibonacci sequence. Create interplay between bass and melody."
#     },
#     {
#         "generate_prompt": "Experimental 7/8 + 4/4 pattern, 174 BPM. Melodic sequence based on Fibonacci intervals. Strategic silences.",
#         "edit_prompt": "Take this loop and add suspended 4th to all major chords. Push rhythm forward by 16th note. Create countermelody in contrary motion."
#     },
#     {
#         "generate_prompt": "West African-inspired 125 BPM. Pentatonic phrases. Emphasize 1 and 3+. Call-response structure between voices.",
#         "edit_prompt": "Take this loop and add 3/4 bar every 12 bars. Create drunken feel by shifting timing. Add countermelody that resolves tension."
#     }
# ]
# %%
