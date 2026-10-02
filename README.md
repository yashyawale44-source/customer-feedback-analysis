# Customer Feedback Analysis & Automated Response

**Imarticus Data Science Internship — Assessment (Python Foundations & Gen AI)**

A Python pipeline that cleans e-commerce reviews, isolates **Critical** reviews with plain rules (no Machine Learning), finds the top complaint keywords, and uses **Generative AI** to draft personalised, empathetic apology emails for the 3 most urgent cases.

## Files

| File | Description |
|---|---|
| `Customer_Feedback_Analysis.ipynb` | Main notebook. All code, charts, insights and the 3 AI-generated emails |
| `Womens Clothing E-Commerce Reviews.csv` | Dataset ([Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)), 23,486 reviews with 1–5★ ratings |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for API keys |
| `outputs/` | Created when the notebook runs: ranked critical reviews and the generated emails (CSV) |

---

## 1. Approach to data cleaning

**Missing values**
| Column | Missing | Action | Reason |
|---|---|---|---|
| `Review Text` | 845 | **Dropped** | No text means nothing to analyse or respond to |
| `Title` | 3,810 | Filled with `""` | Optional field. The review body is still useful |
| `Division / Department / Class Name` | 14 | Filled with `"Unknown"` | Keeps the review and doesn't guess the category |

**Other fixes:** removed exact duplicate reviews (same item + text + rating), fixed the source typo `Initmates` → `Intimates`, and enforced integer types for `Rating` and `Recommended IND`.

**Text cleaning (`clean_text()`):** decode HTML entities, normalise unicode to ASCII, **lowercase**, merge contractions (`didn't` → `didnt`), **remove special characters, digits and punctuation**, and collapse whitespace.
The **original** text is kept for the LLM, which writes better with natural text. The **cleaned** text is used for keyword counting.

## 2. Rule-based logic (no ML)

1. **Critical filter:** `Rating <= 2` (1★ and 2★ reviews).
2. **Complaint keywords (`top_complaint_keywords()`):** tokenise with `str.split()`, remove a hand-written stop-word list (English filler words plus domain-neutral words such as *dress* or *wear*), then count with `collections.Counter`:
   - (a) top single words, (b) top two-word phrases (e.g. *"poor quality"*),
   - (c) **lift**, which is how many times more often a word appears in Critical reviews than in 4–5★ reviews. This surfaces true complaint words instead of words that are common everywhere.
3. **Complaint themes:** keyword dictionaries tag each Critical review with themes such as *Fit / Sizing*, *Quality / Defects* or *Not as pictured*.
4. **Urgency score** (used to choose the 3 reviews for Gen AI):
   `+3` for 1★ (`+2` for 2★) · `+1` if not recommended · `+1` per complaint theme · `+1` per 25 words (max 4) · `+log2(1 + helpful votes)` (max 3).
   The top 3 are taken from **different products**.

## 3. Gen AI outreach

`generate_email()` sends a **system prompt** (the Customer Support Agent persona and rules: empathetic, address each specific complaint, offer a return or refund, never invent order numbers or codes, under 150 words) together with a **user prompt** containing the review, its product category and the detected themes. It works with three providers:

| Priority | Provider | Default model |
|---|---|---|
| 1 | Google Gemini (`GEMINI_API_KEY`) | `gemini-2.5-flash` |
| 2 | OpenAI (`OPENAI_API_KEY`) | `gpt-4o-mini` |
| 3 | HuggingFace local model (no key) | `Qwen/Qwen2.5-1.5B-Instruct` |

---

## How to run

### 1. Install dependencies (Python 3.10+)
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```
> For the local HuggingFace model on a CPU-only machine, a smaller torch build is available:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### 2. Supply an API key (optional but recommended)
Copy `.env.example` to `.env` and fill in **one** key:
```
GEMINI_API_KEY=your-key-here      # free key: https://aistudio.google.com/apikey
# or
OPENAI_API_KEY=sk-...             # https://platform.openai.com/api-keys
```
You can instead set the key as an environment variable:
- Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`
- macOS/Linux: `export GEMINI_API_KEY="your-key"`

Keys are **never written into the notebook**, and `.env` is excluded from the submission (see `.gitignore`). Never share or commit your real `.env`. If no key is found, the notebook automatically uses the free local HuggingFace model. It downloads about 3 GB the first time and takes roughly 1 minute per email on a CPU. To force a provider, set `LLM_PROVIDER=gemini|openai|huggingface`.

### 3. Run the notebook
```bash
jupyter notebook Customer_Feedback_Analysis.ipynb
```
Then choose **Kernel → Restart & Run All**. Keep the CSV in the same folder as the notebook.

---

## Known limitations

- **Truncated reviews.** Review text in this dataset is cut off at about 500 characters (often mid-sentence). The prompt tells the model to address only what is actually written.
- **Word limit.** Emails must be under 150 words. `generate_with_fallback()` checks this and regenerates over-length drafts.
- **Human review required.** The refund/exchange offer in the emails is a template next step, so match it to the company's real return policy. The prompt tells the model not to claim feedback was passed to any team. All drafts should be checked by a person before sending.
- **Reproducibility.** Gemini output varies slightly between runs, so a fresh *Restart & Run All* will produce different email wording from the saved outputs.
