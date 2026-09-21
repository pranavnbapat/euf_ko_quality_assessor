# Understanding KO Quality Assessment (For Non-Technical Users)

## The Four Quality Pillars Explained

We check every KO across four important dimensions. Think of it like inspecting a house:

---

### 1. Structural Quality (30% of total score)

**What it means:** Is the document physically complete and well-formatted?

**Simple analogy:** Like checking if a house has all its walls, roof, doors, and windows in place.

**What we check:**
| Question | Why It Matters |
|----------|----------------|
| Does it have a title? | You can't find a book without a title on the spine |
| Does it have a description? | The back-cover helps you decide if it's relevant |
| Is there actual content? | An empty book is useless |
| Are there enough keywords? | Tags help organize the library |
| Is the text readable? | Garbled text (too many symbols) is hard to read |

**Good Example:** A complete article with title, summary, full text, and 5-10 relevant keywords.

**Bad Example:** A KO with only a title and 2 sentences of content, no description.

**Common Issues (Flags):**
- "Missing description" - No back-cover summary
- "Very short content" - Not enough information provided
- "High noise" - Too many strange symbols or formatting errors

---

### 2. Semantic Quality (35% of total score)

**What it means:** Is the content clear, useful, and meaningful?

**Simple analogy:** Like checking if the house has sensible room layouts, good lighting, and furniture that makes sense together.

**What we check:**
| Question | Why It Matters |
|----------|----------------|
| Is the writing clear? | Short, simple sentences are easier to understand |
| Is it diverse? | Repeating the same words is boring and less informative |
| Does it have substance? | Too many filler words ("the", "and", "very") means less real information |
| Do the parts match? | The title should describe the actual content |
| Is it consistent? | The article shouldn't contradict itself |

**Good Example:** An article titled "Organic Tomato Farming" that actually discusses organic tomato farming methods in clear, varied language with useful details.

**Bad Example:** An article with repetitive phrases, confusing writing, or a title that doesn't match the content (like "Tomato Farming" that talks mostly about potatoes).

**Common Issues (Flags):**
- "Low diversity" - Uses the same words over and over
- "Inconsistent metadata" - Title and content don't match
- "Low information density" - Lots of words, little substance

---

### 3. Functional Quality (25% of total score)

**What it means:** Can people actually find and use this document when searching?

**Simple analogy:** Like checking if a house has a visible address, is on the map, and has good roads leading to it.

**What we check:**
| Question | Why It Matters |
|----------|----------------|
| Can search engines find it? | Important words should appear naturally throughout |
| Can AI systems understand it? | Content should work well with modern search tools |
| Is the vocabulary diverse? | Using different but related terms helps more people find it |
| Is it search-friendly? | The right balance of common and specific terms |

**Good Example:** An article about "sustainable irrigation" that naturally uses related terms like "water conservation," "drip systems," "efficient watering," making it findable through various searches.

**Bad Example:** An article that uses only one technical term repeated 50 times, or one so generic that no search would surface it.

**Common Issues (Flags):**
- "Poor searchability" - Hard to find through searching
- "High repetition" - Same words used too frequently
- "Low embedding yield" - AI systems struggle to categorize it

---

### 4. Domain Quality (10% of total score)

**What it means:** Is this actually about agriculture/farming?

**Simple analogy:** Like checking if a house advertised as a "farmhouse" is actually on a farm and used for farming, not just decorated with rustic furniture.

**What we check:**
| Question | Why It Matters |
|----------|----------------|
| Does it use farming vocabulary? | Words like "crop," "harvest," "soil," "irrigation" |
| Is it relevant to agriculture? | The actual topic should be farming-related |
| Does it match our collection? | We want agricultural content, not random topics |

**Good Example:** An article discussing wheat cultivation, pest management in cornfields, or livestock health - clearly agricultural.

**Bad Example:** An article about general business management or city tourism that happens to mention "farm" once in passing.

**Common Issues (Flags):**
- "Low domain relevance" - Doesn't seem to be about agriculture
- "No anchor matches" - Missing key agricultural terminology

---

## What is MNLI? (Semantic Consistency Check)

**MNLI** stands for "Multi-Genre Natural Language Inference" - but that's just technical jargon.

**What it actually does:** It checks if the **content matches the promise of the title and description**.

**Simple analogy:** Like fact-checking whether a restaurant's menu accurately describes the food they serve.

**How it works:**

Imagine you see a book titled:
> **"Complete Guide to Beekeeping for Beginners"**

But when you open it, the content is about:
> "The history of commercial airlines in the 20th century"

The MNLI check would flag this as **inconsistent** - the content doesn't match the title's promise.

**Another example:**

| Title Promises | Content Delivers | MNLI Result |
|----------------|------------------|-------------|
| "Organic Tomato Farming" | Detailed steps for growing tomatoes organically | ✅ Consistent (High score) |
| "Organic Tomato Farming" | Brief mention of tomatoes, mostly about car repair | ❌ Inconsistent (Low score) |
| "Winter Wheat Diseases" | Comprehensive disease guide for wheat | ✅ Consistent (High score) |
| "Winter Wheat Diseases" | General discussion about all grains, no disease info | ⚠️ Weak consistency (Medium score) |

**Why this matters:**
- **For users:** They don't waste time opening irrelevant documents
- **For the system:** Prevents misleading search results
- **For editors:** Catches mismatches between metadata and content

**What the score means:**
- **High (0.8-1.0):** Content strongly supports the title/description
- **Medium (0.5-0.8):** Content somewhat relates to the title/description
- **Low (0.0-0.5):** Content doesn't match or contradicts the title/description

---

## How to Read a Quality Report

When you get a quality report, here's what to look for:

### Overall Score (0-100)

| Score | Grade | Meaning |
|-------|-------|---------|
| 85-100 | A | Excellent quality - ready to publish |
| 70-84 | B | Good quality - minor improvements possible |
| 55-69 | C | Acceptable - should be improved before publishing |
| 0-54 | D | Poor quality - needs significant work |

### Weighted vs. Unweighted

- **Unweighted (0-100):** Simple sum of all four pillars (each 0-25)
- **Weighted (0-100):** Prioritizes what matters most:
  - Semantic (clarity) counts the most (35%)
  - Structural (completeness) second (30%)
  - Functional (searchability) third (25%)
  - Domain (agricultural relevance) fourth (10%)

### Notes Column

This tells you quickly what's wrong:
- **"Missing description"** - Add a summary
- **"Few keywords"** - Add more tags
- **"Detected non-EN metadata language: nl"** - Content is in Dutch, not English
- **"Content was truncated"** - Article was very long, only analyzed first part

---

## Common Quality Issues and Solutions

| Problem | Which Pillar | Simple Fix |
|---------|--------------|------------|
| No description | Structural | Write a 2-3 sentence summary |
| Very short content | Structural | Expand the article with more details |
| Title doesn't match content | Semantic (MNLI) | Rewrite title to reflect actual content |
| Too repetitive | Semantic | Use synonyms and varied phrasing |
| Hard to find via search | Functional | Include relevant keywords naturally |
| Not obviously about farming | Domain | Use more agricultural terminology |
| Confusing writing | Semantic | Break long sentences into shorter ones |

---

## Summary Table

| Pillar | 30-Second Explanation | Analogy |
|--------|----------------------|---------|
| **Structural** (30%) | Is it complete and properly formatted? | Does the house have all its walls and roof? |
| **Semantic** (35%) | Is it clear, useful, and consistent? | Are the rooms laid out sensibly with good furniture? |
| **Functional** (25%) | Can people find it when searching? | Is the address visible and on the map? |
| **Domain** (10%) | Is it actually about agriculture? | Is the farmhouse actually on a working farm? |
| **MNLI** | Does content match the title's promise? | Does the menu accurately describe the food? |

---

