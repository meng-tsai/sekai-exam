# Sekai Take-Home Challenge

Build a tiny squad of AI agents that learns to recommend the right stories to the right fans—exactly what you’ll do with us in production!

---

## 1 / Mission

Write code that iterates on a recommendation prompt until it consistently picks 10 highly-relevant Sekai stories for a user, given:

- **5 sample Sekai stories (below).**  
  Use these as seed data—expand to ≈ 100 stories with an LLM of your choice.

- **5 full user profiles (below).**  
  Use them as ground truth examples and synthesize additional test users as needed.

You’re free to choose any multi-agent framework, model stack, or hosting setup.

---

## 2 / High-level Flow to Implement

| Agent                | Purpose                                                             | Model Guidance                                                    |
| -------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Prompt-Optimizer     | Proposes prompt tweaks based on prior evaluations.                  | Any model you like.                                               |
| Recommendation Agent | Returns 10 story IDs for a simulated new user. Must be fast.        | Gemini 2.0 Flash (or an equivalently speedy “no-thinking” model). |
| Evaluation Agent     | 1. Reads a full user profile.                                       |
|                      | 2. Simulates the tags that user would tick on Sekai’s first screen. |
|                      | 3. Feeds those simulated tags to the Recommendation Agent.          |
|                      | 4. Computes an eval score against “ground-truth” recommendations.   | Gemini 2.5 Pro or comparable SOTA reasoning-heavy model.          |

Orchestrate them in an autonomous loop:  
**optimize → recommend → evaluate → feedback → repeat**  
until you hit a self-chosen score or time budget.

---

## 3 / Ground Truth & Scoring

- **Ground-truth list**: Ask your Evaluation Agent to recommend directly from the full profile + full story pool.
- **Metric**: Pick something sensible (e.g., precision@10, mean-recall, semantic overlap). Tell us why.
- **Target**: Stop when your metric plateaus or time runs out.

---

## 4 / Data Provided

### 4.1 Sample User Profiles

**USER 1**  
choice-driven, high-agency, dominant protector strategist; underdog, rivalry, team-vs-team, hero-vs-villain, internal-struggle, tournament conflicts; master-servant, royalty-commoner, captor-captive power-dynamics; high-immersion lore-expander, community-engagement; power-fantasy, moral-ambiguity; isekai escapism; romance, forbidden-love, love-triangles, found-family, reverse-harem; enemies-to-lovers, slow-burn; reincarnation, devil-powers, jujitsu-sorcerer; betrayal, loyalty, survival, redemption; fandoms: Naruto, Dragon Ball, Jujutsu-Kaisen, Genshin-Impact, One-Piece, Demon-Slayer, Chainsaw-Man, Marvel/DC; crossover, anti-hero, strategy, fan-groups.

**USER 2**  
Self-insert choice-driven narrator as reluctant/supportive guardian, disguised royalty, rookie competitor. Likes Re:Zero/Naruto/MyHeroAcademia. Prefers cafes, academies, fantasy kingdoms (Konoha, Hogwarts, Teyvat), cities. Genres: supernatural/contemporary/historical romance, fantasy, action, horror. Enjoys supernatural beings, magic/curses/quirks. Favors harem, love triangles, power imbalance, enemies-to-lovers, underdog, redemption. Emotional catalysts: forbidden desires, rival advances, legacy. Content: action, romance.

**USER 3**  
Male roleplayer seeking immersive, choice-driven narratives; self-insert underdog, reluctant hero, dominant protector power fantasy. Prefers one-on-one romance, found-family bonds, intense angst, trauma healing. Loves supernatural—nine-tailed foxes, vampires, magic. Achievement-hunter chasing epic conclusions. Morally flexible exploration sans non-consensual, gore, character death. Co-creative, supportive, detail-rich storytelling. Leaderboard climber, protective sibling loyalty, guilt.

---

### 4.2 Sample Sekai Stories (use as seed data)

- **ID: 217107**  
  **Title**: Stranger Who Fell From The Sky  
  **Intro**: You are Devin, plummeting towards Orario with no memory of how you got here...  
  **Tags**: danmachi, reincarnation, heroic aspirations, mystery origin, teamwork, loyalty, protectiveness

- **ID: 273613**  
  **Title**: Trapped Between Four Anime Legends!  
  **Intro**: You're caught in a dimensional rift with four anime icons. Goku wants to spar...  
  **Tags**: crossover, jujutsu kaisen, dragon ball, naruto, isekai, dimensional travel, reverse harem

- **ID: 235701**  
  **Title**: New Transfer Students vs. Class 1-A Bully  
  **Intro**: You and Zeroku watch in disgust as Bakugo torments Izuku again...  
  **Tags**: my hero academia, challenging authority, bullying, underdog, disruptors

- **ID: 214527**  
  **Title**: Zenitsu Touched Your Sister's WHAT?!  
  **Intro**: Your peaceful afternoon at the Butterfly Estate shatters when Zenitsu accidentally gropes Nezuko...  
  **Tags**: demon slayer, protective instincts, comedic panic, violent reactions

- **ID: 263242**  
  **Title**: Principal's Daughter Dating Contest  
  **Intro**: You are Yuji Itadori, facing off against Tanjiro and Naruto for Ochako's heart...  
  **Tags**: crossover, romantic comedy, forced proximity, harem, dating competition

Feel free to synthesize more stories or users to stress-test your agents.

---

## 5 / Minimum Deliverables

- Git repo (any language) with a one-command demo.
- README or brief doc covering:
  - architecture & agent roles
  - caching strategy (embed & prompt caches)
  - evaluation metric & stopping rule
  - how you’d scale to production volumes
- Log/table showing ≥ 3 optimization cycles with their eval scores.
- ≤ 5-minute Loom/YouTube walk-through (optional but appreciated).

---

## 6 / Review Criteria

- **Architectural clarity** – is the agent loop clean & extensible?
- **Product intuition** – do metric and ground-truth method make sense for Sekai?
- **Code quality & docs** – can we run it in one go?
- **Result quality** – do the final 10 picks actually fit the user? Runtime & token cost?

---

Have fun, iterate boldly, and show us how you think.  
We can’t wait to see your agents in action! 🎉
