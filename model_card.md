# Model Card: Mood Machine

This project has two mood classifiers:

A rule based model in mood_analyzer.py.
An ML model in ml_experiments.py using scikit-learn.

I compared both models.

## 1. Model Overview

**Model type:**  
I used both models.
The ML model changed more when I changed the dataset.
The rule based model needed manual rule updates.

**Intended purpose:**  
The goal is to classify short text into positive, negative, neutral, or mixed.

**How it works (brief):**  
Rule based model:
It preprocesses text into tokens.
It adds points for positive signals.
It subtracts points for negative signals.
It handles negation, emojis, slang, and short text thresholds.
It then maps score to a label.

ML model:
It uses CountVectorizer bag of words features.
It trains LogisticRegression on `SAMPLE_POSTS` and `TRUE_LABELS`.
It predicts labels from learned word patterns.

## 2. Data

**Dataset description:**  
There are 17 posts in `SAMPLE_POSTS`.
I added 6 new posts and matching labels in `TRUE_LABELS`.

**Labeling process:**  
I labeled each post based on the main mood in the sentence.
I used mixed when both positive and negative signals were present.
Hard examples were:
- "I'm dead 😂"
- "Missed the bus again, love that for me"
These can be interpreted in different ways.

**Important characteristics of your dataset:**  
- It contains slang and emojis.
- It includes sarcasm.
- It has mixed-feeling posts.
- It has short and ambiguous posts.

**Possible issues with the dataset:**  
The dataset is still small.
Some labels are subjective.
It may miss language from other communities and styles.

## 3. How the Rule Based Model Works (if used)

**Your scoring rules:**  
The model starts with score 0.
Positive words add points.
Negative words subtract points.
Negation flips nearby sentiment words.
Some emojis and slang use custom weights.
Short texts use a ratio threshold.
Longer texts use stronger score thresholds.
Balanced contrastive text can be labeled mixed.

**Strengths of this approach:**  
It is easy to understand.
It is easy to debug.
It works well on patterns covered by rules.

**Weaknesses of this approach:**  
It can miss new slang.
It can fail on subtle sarcasm.
It needs manual tuning for edge cases.
It can overreact to a few weighted words.

## 4. How the ML Model Works (if used)

**Features used:**  
Bag of words using CountVectorizer.

**Training data:**  
The model trained on `SAMPLE_POSTS` and `TRUE_LABELS`.

**Training behavior:**  
The model changed quickly when I added new labeled examples.
It matched the updated labels on the training dataset.

**Strengths and weaknesses:**  
Strengths:
- It learns patterns automatically.
- It adapted better to my added data.

Weaknesses:
- Reported accuracy is on training data, so it can overfit.
- It can learn spurious word-label shortcuts.

## 5. Evaluation

**How you evaluated the model:**  
I evaluated both models on dataset.py labeled posts.
Rule based accuracy was 0.88.
ML accuracy was 1.00 on the same training dataset.

**Examples of correct predictions:**  
- "Today was a terrible day" -> negative.
It has a strong negative keyword.
- "Highkey relaxed and chill tonight :)" -> positive.
It has positive words and a positive emoji signal.
- "I am tired and excited for tomorrow" -> mixed.
It has both negative and positive words.

**Examples of incorrect predictions:**  
Rule based mistakes after dataset expansion:
- "I'm dead 😂" predicted positive, true mixed.
The laugh emoji got a strong positive weight.
- "Missed the bus again, love that for me" predicted mixed, true negative.
Sarcasm and frustration were not fully captured.

ML model had no errors on the training set in this run.
This does not prove it will generalize to new unseen text.

## 6. Limitations

- The dataset is small.
- Some labels are ambiguous.
- Sarcasm is still hard for both models.
- Rule based performance depends on hand-written rules.
- ML score is measured on training data, not a separate test split.

## 7. Ethical Considerations

- The model can misclassify distress messages.
- It may misread slang from different communities.
- Wrong labels can lead to unfair decisions.
- Personal text analysis raises privacy concerns.

## 8. Ideas for Improvement

- Add more labeled posts with diverse language styles.
- Create a train/validation/test split.
- Add phrase-level sarcasm rules for the rule based model.
- Improve emoji and slang normalization.
- Update explain so it matches weighted scoring exactly.
