"""
Exploratory Data Analysis (EDA) for Chess Games Dataset
========================================================
Analyses:
  1. Move Sequence Lengths — histogram of turns per game
  2. Item Vocabulary Size — unique move tokens
  3. Move Popularity Distribution — opening vs late-game move frequencies
  4. Player Activity Frequency — games per unique player (for FPMC sparsity check)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — plots saved to file only
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ── Load data ────────────────────────────────────────────────────────────────
DATA_PATH = "games_export.csv"
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}\n")

# =============================================================================
# 1. Move Sequence Lengths
# =============================================================================
print("=" * 60)
print("1. MOVE SEQUENCE LENGTHS")
print("=" * 60)

# `turns` column gives the total number of half-moves (plies)
print(df["turns"].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df["turns"], bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Number of Turns (Half-Moves)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Distribution of Game Lengths (Turns)")
axes[0].axvline(df["turns"].median(), color="red", linestyle="--", label=f'Median = {df["turns"].median():.0f}')
axes[0].axvline(df["turns"].mean(), color="orange", linestyle="--", label=f'Mean = {df["turns"].mean():.1f}')
axes[0].legend()

# Box plot to highlight outliers
axes[1].boxplot(df["turns"], orientation="vertical", patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.6))
axes[1].set_ylabel("Number of Turns")
axes[1].set_title("Box Plot of Game Lengths")

plt.tight_layout()
plt.savefig("eda_1_move_sequence_lengths.png", dpi=150, bbox_inches="tight")
plt.show()

# Flag very short and very long games
short_games = df[df["turns"] <= 4]
long_games = df[df["turns"] >= 150]
print(f"\nVery short games (<=4 turns): {len(short_games)}")
print(f"Very long games  (>=150 turns): {len(long_games)}")

# =============================================================================
# 2. Item Vocabulary Size
# =============================================================================
print("\n" + "=" * 60)
print("2. ITEM VOCABULARY SIZE")
print("=" * 60)

# Tokenise every move string
all_moves = df["moves"].str.split().explode()
total_tokens = len(all_moves)
unique_moves = all_moves.nunique()

print(f"Total move tokens across all games : {total_tokens:,}")
print(f"Unique move tokens (vocabulary size): {unique_moves:,}")

# Show the most common moves
move_counts = all_moves.value_counts()
print(f"\nTop 20 most frequent moves:")
print(move_counts.head(20).to_string())

fig, ax = plt.subplots(figsize=(12, 5))
top_n = 30
move_counts.head(top_n).plot(kind="bar", ax=ax, color="#55A868", edgecolor="white")
ax.set_xlabel("Move Token")
ax.set_ylabel("Frequency")
ax.set_title(f"Top {top_n} Most Frequent Move Tokens (Vocabulary = {unique_moves:,})")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_2_vocabulary_top_moves.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# 3. Move Popularity Distribution (Opening vs Late-Game)
# =============================================================================
print("\n" + "=" * 60)
print("3. MOVE POPULARITY DISTRIBUTION")
print("=" * 60)

# ---- 3a. White's first move frequency ----
first_white_move = df["moves"].str.split().str[0]  # move at index 0
first_move_freq = first_white_move.value_counts()

print("White's first move distribution:")
print(first_move_freq.head(15).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of opening moves
first_move_freq.head(15).plot(kind="bar", ax=axes[0], color="#C44E52", edgecolor="white")
axes[0].set_xlabel("White's First Move")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Top 15 White Opening Moves")
axes[0].tick_params(axis="x", rotation=45)

# ---- 3b. Move position distribution (early vs mid vs late) ----
# For each game, tag each move with its position index, then compare frequencies
def get_phase_moves(moves_series, phase):
    """Extract moves belonging to a given phase (opening/middle/endgame)."""
    result = []
    for move_str in moves_series.dropna():
        tokens = move_str.split()
        n = len(tokens)
        if phase == "opening":
            result.extend(tokens[:10])        # first 10 half-moves
        elif phase == "middle":
            mid_start = min(10, n)
            mid_end = max(mid_start, n - 10)
            result.extend(tokens[mid_start:mid_end])
        elif phase == "endgame":
            result.extend(tokens[max(0, n - 10):])
    return Counter(result)

opening_freq = get_phase_moves(df["moves"], "opening")
middle_freq = get_phase_moves(df["moves"], "middle")
endgame_freq = get_phase_moves(df["moves"], "endgame")

# Compare top moves across phases
top_k = 15
phases = {"Opening (first 10 moves)": opening_freq,
          "Middlegame": middle_freq,
          "Endgame (last 10 moves)": endgame_freq}

phase_data = {}
for phase_name, freq in phases.items():
    total = sum(freq.values())
    phase_data[phase_name] = {m: c / total for m, c in freq.most_common(top_k)}

# Plot endgame top moves as example contrast
endgame_top = pd.Series(dict(endgame_freq.most_common(top_k)))
endgame_top.plot(kind="bar", ax=axes[1], color="#8172B2", edgecolor="white")
axes[1].set_xlabel("Move Token")
axes[1].set_ylabel("Frequency")
axes[1].set_title(f"Top {top_k} Endgame Moves (Last 10 Half-Moves)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("eda_3_move_popularity.png", dpi=150, bbox_inches="tight")
plt.show()

# Log-frequency plot to visualise long-tail
print("\nMove frequency distribution (log-log) — checking long-tail effect:")
fig, ax = plt.subplots(figsize=(8, 5))
ranks = np.arange(1, len(move_counts) + 1)
ax.loglog(ranks, move_counts.values, linewidth=1.5, color="#4C72B0")
ax.set_xlabel("Rank (log)")
ax.set_ylabel("Frequency (log)")
ax.set_title("Move Token Frequency — Zipf-style Long-Tail Plot")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("eda_3_move_zipf.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# 4. Player Activity Frequency
# =============================================================================
print("\n" + "=" * 60)
print("4. PLAYER ACTIVITY FREQUENCY")
print("=" * 60)

# Combine white_id and black_id into a single player activity count
white_counts = df["white_id"].value_counts()
black_counts = df["black_id"].value_counts()
player_games = white_counts.add(black_counts, fill_value=0).astype(int).sort_values(ascending=False)

total_players = len(player_games)
print(f"Total unique players: {total_players:,}")
print(f"\nGames per player statistics:")
print(player_games.describe())

# Percentiles
for pct in [50, 75, 90, 95, 99]:
    val = np.percentile(player_games.values, pct)
    print(f"  {pct}th percentile: {val:.0f} games")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of games per player
axes[0].hist(player_games.values, bins=100, color="#DD8452", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("Number of Games Played")
axes[0].set_ylabel("Number of Players")
axes[0].set_title("Player Activity Distribution")
axes[0].set_yscale("log")

# CDF
sorted_vals = np.sort(player_games.values)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
axes[1].plot(sorted_vals, cdf, color="#4C72B0", linewidth=2)
axes[1].set_xlabel("Number of Games Played")
axes[1].set_ylabel("Cumulative Proportion of Players")
axes[1].set_title("CDF of Player Activity")
axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
axes[1].axhline(0.9, color="gray", linestyle=":", alpha=0.5, label="90%")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("eda_4_player_activity.png", dpi=150, bbox_inches="tight")
plt.show()

# Sparsity analysis for FPMC
one_game_players = (player_games == 1).sum()
few_game_players = (player_games <= 3).sum()
print(f"\nPlayers with only 1 game : {one_game_players} ({one_game_players / total_players * 100:.1f}%)")
print(f"Players with <=3 games   : {few_game_players} ({few_game_players / total_players * 100:.1f}%)")
print(f"Players with >10 games  : {(player_games > 10).sum()} ({(player_games > 10).sum() / total_players * 100:.1f}%)")

# Top 10 most active players
print(f"\nTop 10 most active players:")
print(player_games.head(10).to_string())

print("\n" + "=" * 60)
print("EDA COMPLETE — Plots saved as eda_*.png")
print("=" * 60)


# #############################################################################
#                        DATA CLEANING / PREPROCESSING
# #############################################################################
import re

# ── Configurable thresholds ──────────────────────────────────────────────────
MIN_MOVES = 10          # Minimum number of half-moves to keep a game
RARE_MOVE_THRESHOLD = 5  # Moves appearing fewer than k times -> <UNK>

# =============================================================================
# 5. Remove Trivial Games
# =============================================================================
print("\n" + "=" * 60)
print("5. REMOVE TRIVIAL GAMES")
print("=" * 60)

before_count = len(df)
df_clean = df[df["turns"] >= MIN_MOVES].copy()
after_count = len(df_clean)
removed = before_count - after_count

print(f"Threshold          : games with < {MIN_MOVES} half-moves removed")
print(f"Before             : {before_count:,} games")
print(f"After              : {after_count:,} games")
print(f"Removed            : {removed:,} games ({removed / before_count * 100:.1f}%)")

# Show what was removed
removed_games = df[df["turns"] < MIN_MOVES]
print(f"\nRemoved games — victory_status breakdown:")
print(removed_games["victory_status"].value_counts().to_string())

# =============================================================================
# 6. Handle Check / Mate Notation
# =============================================================================
print("\n" + "=" * 60)
print("6. HANDLE CHECK / MATE NOTATION")
print("=" * 60)

# Count how many tokens have + or # before stripping
sample_moves = df_clean["moves"].str.split().explode()
check_tokens = sample_moves.str.contains(r"[+#]", regex=True).sum()
print(f"Tokens with + or # before cleaning: {check_tokens:,} "
      f"({check_tokens / len(sample_moves) * 100:.1f}% of all tokens)")

# Strip trailing + and # from every move token
#   e.g.  Qxf7+  -> Qxf7
#          Qd8#  -> Qd8
#         Bxg7+  -> Bxg7
df_clean["moves"] = df_clean["moves"].apply(
    lambda s: " ".join(tok.rstrip("+#") for tok in s.split())
)

# Verify
sample_after = df_clean["moves"].str.split().explode()
remaining = sample_after.str.contains(r"[+#]", regex=True).sum()
print(f"Tokens with + or # after  cleaning: {remaining:,}")

# Show vocabulary change
vocab_before = sample_moves.nunique()
vocab_after = sample_after.nunique()
print(f"\nVocabulary before stripping: {vocab_before:,}")
print(f"Vocabulary after  stripping: {vocab_after:,}")
print(f"Reduction                  : {vocab_before - vocab_after:,} tokens merged")

# =============================================================================
# 7. Vocabulary Pruning  (<UNK> replacement)
# =============================================================================
print("\n" + "=" * 60)
print(f"7. VOCABULARY PRUNING  (k = {RARE_MOVE_THRESHOLD})")
print("=" * 60)

# Build global frequency table on the cleaned moves
all_clean_moves = df_clean["moves"].str.split().explode()
move_freq = all_clean_moves.value_counts()

rare_moves = set(move_freq[move_freq < RARE_MOVE_THRESHOLD].index)
common_moves = set(move_freq[move_freq >= RARE_MOVE_THRESHOLD].index)

print(f"Total unique moves (post-notation cleaning): {len(move_freq):,}")
print(f"Common moves (freq >= {RARE_MOVE_THRESHOLD})               : {len(common_moves):,}")
print(f"Rare moves  (freq <  {RARE_MOVE_THRESHOLD}) -> <UNK>        : {len(rare_moves):,}")

# Show some examples of rare moves
rare_examples = move_freq[move_freq < RARE_MOVE_THRESHOLD].head(20)
print(f"\nExamples of rare moves (first 20):")
print(rare_examples.to_string())

# Replace rare moves with <UNK>
def replace_rare(move_string, rare_set):
    """Replace any token in rare_set with <UNK>."""
    tokens = move_string.split()
    return " ".join("<UNK>" if tok in rare_set else tok for tok in tokens)

df_clean["moves"] = df_clean["moves"].apply(lambda s: replace_rare(s, rare_moves))

# Final vocabulary stats
final_moves = df_clean["moves"].str.split().explode()
final_vocab = final_moves.nunique()
unk_count = (final_moves == "<UNK>").sum()

print(f"\nFinal vocabulary size (incl. <UNK>): {final_vocab:,}")
print(f"<UNK> token count in corpus        : {unk_count:,} "
      f"({unk_count / len(final_moves) * 100:.2f}% of all tokens)")

# Visualise the pruned vocabulary
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before vs after vocab comparison
labels = ["Before\nCleaning", "After Notation\nStripping", "After Vocab\nPruning"]
sizes = [vocab_before, vocab_after, final_vocab]
colors = ["#C44E52", "#DD8452", "#55A868"]
axes[0].bar(labels, sizes, color=colors, edgecolor="white", width=0.5)
axes[0].set_ylabel("Vocabulary Size")
axes[0].set_title("Vocabulary Size Across Cleaning Steps")
for i, v in enumerate(sizes):
    axes[0].text(i, v + 30, f"{v:,}", ha="center", fontweight="bold")

# Distribution of <UNK> per game
unk_per_game = df_clean["moves"].apply(lambda s: s.split().count("<UNK>"))
axes[1].hist(unk_per_game[unk_per_game > 0], bins=50,
             color="#8172B2", edgecolor="white", alpha=0.85)
axes[1].set_xlabel("Number of <UNK> Tokens per Game")
axes[1].set_ylabel("Number of Games")
axes[1].set_title(f"<UNK> Token Distribution (games with >=1 <UNK>: "
                   f"{(unk_per_game > 0).sum():,})")

plt.tight_layout()
plt.savefig("eda_5_preprocessing_summary.png", dpi=150, bbox_inches="tight")
plt.show()

# =============================================================================
# Save cleaned dataset
# =============================================================================
OUTPUT_PATH = "games_cleaned.csv"
df_clean.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print(f"CLEANED DATASET SAVED -> {OUTPUT_PATH}")
print(f"  Rows   : {len(df_clean):,}")
print(f"  Vocab  : {final_vocab:,} (incl. <UNK>)")
print("=" * 60)
