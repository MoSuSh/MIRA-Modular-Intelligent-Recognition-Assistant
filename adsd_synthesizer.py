import os
import random
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from scipy import signal
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# ADSD SYNTHESIZER FUNCTION (your working version, exactly as you had it)
# =============================================================================
def create_adsd_speech(input_path, output_dir,
                       num_breaks=3,
                       total_break_sec=0.6,
                       breathiness=0.2,
                       strain=0.5,
                       pitch_breaks=2,
                       tremor_depth=0.04,
                       tremor_freq=5.5,
                       speed_slowdown=0.5,
                       random_seed=None):
    """
    Synthesize ADSD speech with controllable parameters.

    New parameter:
        speed_slowdown (float): 0-1, amount of speed reduction at the start of a break
                                (linearly returns to normal by the end).
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    print(f"\n🔊 ADSD Synthesizer (with Speed Variation)")
    print("=" * 50)
    print(f"Settings: breaks={num_breaks} ({total_break_sec}s total), breathiness={breathiness}, strain={strain}, pitch_breaks={pitch_breaks}, tremor_depth={tremor_depth}, speed_slowdown={speed_slowdown}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load audio
    audio, sr = librosa.load(input_path, sr=16000)
    duration = len(audio) / sr
    print(f"📂 Loaded: {os.path.basename(input_path)} ({duration:.2f}s)")

    # --- Vowel detection for placing effects ---
    hop_length = int(0.010 * sr)
    frame_length = int(0.025 * sr)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    pitches, _ = librosa.piptrack(y=audio, sr=sr, win_length=frame_length, hop_length=hop_length)
    voiced = np.max(pitches, axis=0) > 0
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-10)
    vowel_frames = voiced & (rms_norm > 0.3)

    vowel_mask = np.zeros(len(audio), dtype=bool)
    for i, is_vowel in enumerate(vowel_frames):
        if is_vowel:
            start = i * hop_length
            end = min(start + frame_length, len(audio))
            vowel_mask[start:end] = True
    vowel_indices = np.where(vowel_mask)[0]

    if len(vowel_indices) == 0:
        print("⚠️  No vowel regions detected, using entire audio.")
        vowel_indices = np.arange(len(audio))

    output = audio.copy()
    # Keep a list of break positions (in samples) for later strain envelope
    break_sample_ranges = []  # list of (start, end) after any shifts

    # --- 1. Tremor ---
    if tremor_depth > 0:
        t = np.arange(len(audio)) / sr
        tremor = 1 + tremor_depth * np.sin(2 * np.pi * tremor_freq * t)
        output *= tremor
        print(f"   Tremor: {tremor_freq} Hz, depth {tremor_depth}")

    # --- 2. Voice breaks with speed variation ---
    # Each break will have a random duration that sums to total_break_sec
    break_durations = []
    remaining = total_break_sec
    for i in range(num_breaks):
        if i == num_breaks - 1:
            dur = remaining
        else:
            dur = random.uniform(0.5 * total_break_sec/num_breaks, 1.5 * total_break_sec/num_breaks)
            dur = min(dur, remaining)
        break_durations.append(max(0.03, dur))  # at least 30 ms
        remaining -= dur
        if remaining <= 0:
            break

    break_durations = [d for d in break_durations if d > 0]
    num_breaks = len(break_durations)

    min_gap = 0.3  # minimum seconds between breaks
    break_positions_sec = []  # store center times in seconds (updated after shifts)

    # We'll build the output incrementally, because speed changes alter lengths
    output_segments = []
    current_sample = 0

    # First, we need to decide break locations in the original audio (before any shifts)
    # We'll collect them in order.
    break_starts_orig = []
    break_ends_orig = []
    break_centers_orig = []  # sample index of center

    # Use current output (with tremor) as base for placement.
    for dur in break_durations:
        attempts = 0
        placed = False
        while attempts < 200 and not placed:
            idx = np.random.choice(vowel_indices)
            time_sec = idx / sr
            if all(abs(time_sec - pos) > min_gap for pos in break_positions_sec):
                break_positions_sec.append(time_sec)
                n = int(dur * sr)
                start = max(0, idx - n//2)
                end = min(len(output), start + n)
                break_starts_orig.append(start)
                break_ends_orig.append(end)
                break_centers_orig.append(idx)
                placed = True
            attempts += 1

    if len(break_starts_orig) == 0:
        print("⚠️  Could not place any voice breaks.")
    else:
        # Sort breaks by start
        sort_idx = np.argsort(break_starts_orig)
        break_starts_orig = [break_starts_orig[i] for i in sort_idx]
        break_ends_orig = [break_ends_orig[i] for i in sort_idx]
        break_centers_orig = [break_centers_orig[i] for i in sort_idx]

        # Reconstruct output, processing breaks in order
        last_end = 0
        new_break_ranges = []  # will store (start, end) in final output

        for i, (start_orig, end_orig, center_orig) in enumerate(zip(break_starts_orig, break_ends_orig, break_centers_orig)):
            # Add the segment before this break
            if start_orig > last_end:
                output_segments.append(output[last_end:start_orig])
            # Extract the break segment
            seg = output[start_orig:end_orig].copy()
            seg_len = len(seg)

            # --- Speed variation on this segment ---
            if speed_slowdown > 0:
                # Speed factor: linearly increases from min_speed to 1.0
                min_speed = max(0.4, 1.0 - speed_slowdown * 0.6)  # at most 60% slower
                speed_profile = np.linspace(min_speed, 1.0, seg_len)

                # Cumulative output time for each input sample
                # The output time increment for input sample i is 1/speed_profile[i]
                output_time = np.cumsum(1.0 / speed_profile)
                total_output_len = output_time[-1]  # in samples

                # New sample positions (uniform in output time)
                new_len = int(np.round(total_output_len))
                if new_len > 0 and new_len != seg_len:
                    # Interpolate original samples onto the new time grid
                    orig_idx = np.arange(seg_len)
                    new_idx = np.linspace(0, output_time[-1], new_len)
                    # Use linear interpolation
                    seg_speed = np.interp(new_idx, output_time, seg)
                    # Apply fade to avoid clicks at boundaries
                    fade_len = min(int(0.005*sr), new_len//4)
                    if fade_len > 0:
                        fade_in = np.linspace(0, 1, fade_len)
                        fade_out = np.linspace(1, 0, fade_len)
                        seg_speed[:fade_len] *= fade_in
                        seg_speed[-fade_len:] *= fade_out
                    seg = seg_speed
                # else if new_len == 0 or no change, keep seg (shouldn't happen)

            # Attenuate the middle part (simulate aphonia)
            fade_len = min(int(0.005*sr), len(seg)//4)
            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                seg[:fade_len] *= fade_in
                seg[-fade_len:] *= fade_out
            if len(seg) > 2*fade_len:
                seg[fade_len:-fade_len] *= 0.1

            output_segments.append(seg)
            # Record new range (cumulative length)
            new_start = sum(len(s) for s in output_segments) - len(seg)
            new_end = new_start + len(seg)
            new_break_ranges.append((new_start, new_end))
            last_end = end_orig

        # Add the final segment after the last break
        if last_end < len(output):
            output_segments.append(output[last_end:])

        # Concatenate all parts
        output = np.concatenate(output_segments)

        # Update break positions for later use (strain envelope)
        break_sample_ranges = new_break_ranges
        break_positions_sec = [(start+end)/2/sr for (start, end) in new_break_ranges]

        print(f"   Voice breaks: {num_breaks} events, total original duration {total_break_sec:.2f}s (output length may vary due to speed variation)")

    # --- 3. Pitch breaks ---
    pitch_break_mask = np.zeros(len(output), dtype=bool)
    if pitch_breaks > 0 and len(vowel_indices) > 0:
        pitch_shift_range = (-2.5, 2.5)  # semitones
        shift_duration = 0.1  # 100 ms each
        for _ in range(pitch_breaks):
            # Choose a vowel region from original indices, map roughly to new output
            idx_orig = np.random.choice(vowel_indices)
            scale = len(output) / len(audio) if len(audio) > 0 else 1.0
            idx = int(idx_orig * scale)
            idx = max(0, min(len(output)-1, idx))
            dur = shift_duration
            n = int(dur * sr)
            start = max(0, idx - n//2)
            end = min(len(output), start + n)
            seg = output[start:end]
            if len(seg) > sr*0.02:
                shift = random.uniform(*pitch_shift_range)
                shifted = librosa.effects.pitch_shift(y=seg, sr=sr, n_steps=shift)
                # Crossfade
                fade_len = min(int(0.005*sr), n//4)
                if fade_len > 0:
                    fade_in = np.linspace(0, 1, fade_len)
                    fade_out = np.linspace(1, 0, fade_len)
                    shifted[:fade_len] *= fade_in
                    shifted[-fade_len:] *= fade_out
                output[start:end] = shifted
                pitch_break_mask[start:end] = True
        print(f"   Pitch breaks: {pitch_breaks}")

    # --- 4. Breathiness ---
    if breathiness > 0:
        noise = np.random.normal(0, 1, len(output))
        b, a = signal.butter(2, 3000/(sr/2), btype='low')
        noise = signal.filtfilt(b, a, noise)
        noise_level = breathiness * 0.02 * np.abs(output)
        output += noise * noise_level
        print(f"   Breathiness: {breathiness}")

    # --- 5. Strain (hyperadduction) ---
    if strain > 0:
        b, a = signal.butter(2, 2000/(sr/2), btype='high')
        filtered = signal.filtfilt(b, a, output)
        strain_envelope = np.ones_like(output) * strain * 0.3
        # Boost around break positions (using updated break_sample_ranges)
        for (start, end) in break_sample_ranges:
            center = (start + end) // 2
            width = int(0.2 * sr)
            x = np.arange(len(output))
            gauss = np.exp(-0.5 * ((x - center) / (width/3))**2)
            strain_envelope += strain * 0.5 * gauss
        output = output * (1 - strain_envelope) + filtered * strain_envelope
        output = np.tanh(output * 1.1) / 1.1
        print(f"   Strain: {strain}")

    # --- Normalize ---
    max_val = np.max(np.abs(output))
    if max_val > 0.95:
        output = output / max_val * 0.95

    # Save
    original_name = Path(input_path).stem
    out_filename = f"{original_name}_synthesized.wav"
    out_path = Path(output_dir) / out_filename
    sf.write(out_path, output, sr)

    # Save parameters
    params_file = Path(output_dir) / f"{original_name}_synthesized_params.txt"
    with open(params_file, 'w', encoding='utf-8') as f:
        f.write("ADSD Synthesis Parameters\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"num_breaks: {num_breaks}\n")
        f.write(f"total_break_sec: {total_break_sec}\n")
        f.write(f"breathiness: {breathiness}\n")
        f.write(f"strain: {strain}\n")
        f.write(f"pitch_breaks: {pitch_breaks}\n")
        f.write(f"tremor_depth: {tremor_depth}\n")
        f.write(f"tremor_freq: {tremor_freq}\n")
        f.write(f"speed_slowdown: {speed_slowdown}\n")

    print(f"\n✅ Saved to {out_path}")
    print("=" * 50)
    return out_path


# =============================================================================
# BATCH PROCESSING CONFIGURATION
# =============================================================================

MALE_INPUT = r"E:\Dataset\Male"
FEMALE_INPUT = r"E:\Dataset\Female"
OUTPUT_BASE = r"E:\Dataset\Synthesized"
NUM_VERSIONS = 3

# Parameter ranges for randomization (you can tweak these)
PARAM_RANGES = {
    "num_breaks": (4, 7),               # integer range
    "total_break_sec": (2, 4),
    "breathiness": (0.8, 1),
    "strain": (0.5, 0.8),
    "pitch_breaks": (3, 6),              # integer range
    "tremor_depth": (0.4, 0.9),
    "tremor_freq": (4.0, 7.0),
    "speed_slowdown": (0.3, 0.6)
}

def random_params(seed=None):
    """Generate a random parameter set within the defined ranges."""
    if seed is not None:
        random.seed(seed)
    params = {}
    for key, (low, high) in PARAM_RANGES.items():
        if key in ["num_breaks", "pitch_breaks"]:
            params[key] = random.randint(low, high)
        else:
            params[key] = round(random.uniform(low, high), 3)
    return params

def process_gender(input_dir, output_dir, gender_prefix):
    """Process all .wav files in input_dir, generating NUM_VERSIONS each."""
    wav_files = list(Path(input_dir).glob("*.wav"))
    print(f"\n📁 Processing {gender_prefix} files: {len(wav_files)} found")
    
    for wav_path in tqdm(wav_files, desc=gender_prefix):
        base_name = wav_path.stem   # e.g., "male_1"
        
        # Locate corresponding transcript (must have same name, .txt extension)
        txt_path = wav_path.with_suffix('.txt')
        if not txt_path.exists():
            print(f"⚠️  Transcript not found for {wav_path.name}, skipping.")
            continue
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        
        # Generate NUM_VERSIONS synthesized versions
        for version in range(1, NUM_VERSIONS + 1):
            # Use a deterministic seed based on file name and version
            seed = hash(f"{base_name}_v{version}") % (2**32)
            params = random_params(seed)
            
            # Define output file names
            synth_wav = Path(output_dir) / f"{base_name}_synth{version}.wav"
            synth_txt = Path(output_dir) / f"{base_name}_synth{version}.txt"
            
            # Skip if already exists (allows resuming)
            if synth_wav.exists() and synth_txt.exists():
                continue
            
            # Create a temporary directory inside output_dir for this file
            temp_dir = Path(output_dir) / "temp_synthesis"
            temp_dir.mkdir(exist_ok=True)
            
            try:
                # Call the synthesizer – it saves a file with "_synthesized.wav" suffix
                create_adsd_speech(
                    str(wav_path),
                    str(temp_dir),
                    num_breaks=params["num_breaks"],
                    total_break_sec=params["total_break_sec"],
                    breathiness=params["breathiness"],
                    strain=params["strain"],
                    pitch_breaks=params["pitch_breaks"],
                    tremor_depth=params["tremor_depth"],
                    tremor_freq=params["tremor_freq"],
                    speed_slowdown=params["speed_slowdown"],
                    random_seed=seed
                )
                
                # The generated file will be named f"{base_name}_synthesized.wav"
                generated = temp_dir / f"{base_name}_synthesized.wav"
                if generated.exists():
                    # Move and rename to desired name
                    generated.rename(synth_wav)
                    # Write the transcript
                    with open(synth_txt, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                else:
                    print(f"❌ Synthesis failed for {wav_path.name} version {version} – output file not found.")
            except Exception as e:
                print(f"❌ Error processing {wav_path.name} version {version}: {e}")
            
            # Clean up temporary directory (remove all contents)
            for item in temp_dir.iterdir():
                item.unlink()
            temp_dir.rmdir()

# Create output directories
male_out = Path(OUTPUT_BASE) / "Male"
female_out = Path(OUTPUT_BASE) / "Female"
male_out.mkdir(parents=True, exist_ok=True)
female_out.mkdir(parents=True, exist_ok=True)

# Process both genders
process_gender(Path(MALE_INPUT), male_out, "Male")
process_gender(Path(FEMALE_INPUT), female_out, "Female")

print("\n✅ Batch synthesis complete!")
print(f"👨 Male files output: {male_out}")
print(f"👩 Female files output: {female_out}")