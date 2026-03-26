# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()

        # Replace a few simple emoticons/emojis with explicit tokens so they
        # survive punctuation removal and are handled as separate features.
        emoji_map = {
          ":)": " emoji_smile ",
          ":-(": " emoji_sad ",
          "🥲": " emoji_tear_smile ",
          "😂": " emoji_laugh ",
        }
        for symbol, replacement in emoji_map.items():
          cleaned = cleaned.replace(symbol, replacement)

        # Remove punctuation and symbols (keep letters, numbers, and spaces).
        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        tokens = cleaned.split()

        # Normalize character runs sooooo -> soo (keep up to two repeats).
        normalized_tokens = [re.sub(r"([a-z])\1{2,}", r"\1\1", token) for token in tokens]

        # Guard against empty tokens if input had unusual spacing/symbols.
        return [token for token in normalized_tokens if token]

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        tokens = self.preprocess(text)
        score = 0

        # Enhancement 1: simple negation handling for adjacent sentiment words.
        negation_words = {"not", "never", "no"}

        # Enhancement 2: treat emoji/slang tokens as stronger signals.
        signal_weights = {
          "emoji_smile": 2,
          "emoji_laugh": 2,
          "emoji_sad": -2,
          "emoji_tear_smile": -1,
          "lol": 1,
          "lmao": 1,
          "rofl": 1,
          "dead": -1,
          "mood": 1,
          "lowkey": -1,
          "highkey": 1,
          # Common frustration context words (helps with light sarcasm).
          "stuck": -2,
          "traffic": -1,
        }

        for i, token in enumerate(tokens):
          # First, apply explicit weighted signals if present.
          if token in signal_weights:
            score += signal_weights[token]
            continue

          is_negated = i > 0 and tokens[i - 1] in negation_words

          # Then apply base word-list scoring.
          if token in self.positive_words:
            score += -1 if is_negated else 1
          elif token in self.negative_words:
            score += 1 if is_negated else -1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        tokens = self.preprocess(text)
        token_count = len(tokens)
        score = self.score_text(text)
        has_contrast_cue = "but" in tokens or "or" in tokens
        has_positive_word = any(token in self.positive_words for token in tokens)
        has_negative_word = any(token in self.negative_words for token in tokens)

        if score == 0:
          # Balanced positive/negative language is usually better represented
          # as mixed rather than neutral.
          if has_contrast_cue or (has_positive_word and has_negative_word):
            return "mixed"
          return "neutral"

        # For short text, use score density (percent of token length) so
        # very short messages are not over-penalized or over-boosted.
        if token_count <= 6:
          score_ratio = score / max(token_count, 1)
          short_threshold = 0.1
          if score_ratio >= short_threshold:
            return "positive"
          if score_ratio <= -short_threshold:
            return "negative"
          return "mixed"

        # For longer text, contrast words often indicate mixed/ambivalent tone
        # when the score is only slightly polarized.
        if has_contrast_cue and abs(score) <= 2:
          return "mixed"

        # For normal length text, keep a fixed stronger threshold.
        if score >= 2:
          return "positive"
        if score <= -2:
          return "negative"
        return "mixed"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
