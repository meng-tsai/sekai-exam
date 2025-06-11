import os
import json
from pathlib import Path

import openai
from dotenv import load_dotenv

from sekai_optimizer.data.types import Dataset


def _get_generation_prompt() -> str:
    """Returns the system prompt for generating synthetic data."""
    # This prompt is based on Docs/goal.md and Docs/engineering_design_doc.md
    return """
You are a data synthesis expert for a content recommendation engine.
Your task is to generate a dataset of approximately 100 fictional stories and 50 diverse user profiles based on the provided seed examples.

---
**SEED STORY EXAMPLES**

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
---
**SEED USER PROFILE EXAMPLES**

**USER 1**
choice-driven, high-agency, dominant protector strategist; underdog, rivalry, team-vs-team, hero-vs-villain, internal-struggle, tournament conflicts; master-servant, royalty-commoner, captor-captive power-dynamics; high-immersion lore-expander, community-engagement; power-fantasy, moral-ambiguity; isekai escapism; romance, forbidden-love, love-triangles, found-family, reverse-harem; enemies-to-lovers, slow-burn; reincarnation, devil-powers, jujitsu-sorcerer; betrayal, loyalty, survival, redemption; fandoms: Naruto, Dragon Ball, Jujutsu-Kaisen, Genshin-Impact, One-Piece, Demon-Slayer, Chainsaw-Man, Marvel/DC; crossover, anti-hero, strategy, fan-groups.

**USER 2**
Self-insert choice-driven narrator as reluctant/supportive guardian, disguised royalty, rookie competitor. Likes Re:Zero/Naruto/MyHeroAcademia. Prefers cafes, academies, fantasy kingdoms (Konoha, Hogwarts, Teyvat), cities. Genres: supernatural/contemporary/historical romance, fantasy, action, horror. Enjoys supernatural beings, magic/curses/quirks. Favors harem, love triangles, power imbalance, enemies-to-lovers, underdog, redemption. Emotional catalysts: forbidden desires, rival advances, legacy. Content: action, romance.

**USER 3**
Male roleplayer seeking immersive, choice-driven narratives; self-insert underdog, reluctant hero, dominant protector power fantasy. Prefers one-on-one romance, found-family bonds, intense angst, trauma healing. Loves supernatural—nine-tailed foxes, vampires, magic. Achievement-hunter chasing epic conclusions. Morally flexible exploration sans non-consensual, gore, character death. Co-creative, supportive, detail-rich storytelling. Leaderboard climber, protective sibling loyalty, guilt.
---

**INSTRUCTIONS:**

1.  **Stories:**
    *   Generate a list of ~100 stories.
    *   Each story must have an `id` (unique integer), a `title` (string), `intro` (string), and `tags` (a list of strings).
    *   The stories should be inspired by the themes and style of the seed examples.
    *   Ensure a good variety of tags across the stories, mixing and matching themes.

2.  **User Profiles:**
    *   Generate a list of ~50 user profiles.
    *   Each user must have a `user_id` (unique integer), `name` (string), and a detailed `profile` (string).
    *   The profiles should describe a user's preferences in a similar manner to the seed examples.
    *   Create diverse and sometimes conflicting profiles to ensure the recommendation model is well-tested.

3.  **Output Format:**
    *   You MUST return a single JSON object.
    *   The JSON object must have two keys: "stories" and "users".
    *   The value for each key should be a list of the corresponding objects described above.
    *   Do not include any text outside of the main JSON object.
"""


def synthesize_and_save_data(
    client: openai.OpenAI,
    stories_filepath: Path,
    users_filepath: Path,
) -> None:
    """
    Generates synthetic story and user data using an LLM and saves it to files.

    Args:
        client: An initialized OpenAI client.
        stories_filepath: The path to save the stories JSON file.
        users_filepath: The path to save the users JSON file.
    """
    print("Generating synthetic data using GPT-4o with Pydantic schema enforcement...")
    system_prompt = _get_generation_prompt()

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Please generate the dataset now."},
        ],
        response_format=Dataset,
        temperature=0.7,
    )

    # The `parse` method automatically validates and returns the Pydantic model
    parsed_response = completion.choices[0].message.parsed
    if not parsed_response:
        refusal = completion.choices[0].message.refusal
        raise ValueError(f"LLM refused to generate data. Refusal: {refusal}")

    print("Successfully received and parsed response from LLM.")

    # Convert Pydantic models to dictionaries for JSON serialization
    stories = [story.model_dump() for story in parsed_response.stories]
    users = [user.model_dump() for user in parsed_response.users]

    # Ensure parent directories exist
    stories_filepath.parent.mkdir(parents=True, exist_ok=True)
    users_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save the data to the specified files
    with open(stories_filepath, "w") as f:
        json.dump(stories, f, indent=2)
    print(f"Successfully saved {len(stories)} stories to {stories_filepath}")

    with open(users_filepath, "w") as f:
        json.dump(users, f, indent=2)
    print(f"Successfully saved {len(users)} users to {users_filepath}")


if __name__ == "__main__":
    # This allows the script to be run directly to generate the data
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY not found in .env file or environment variables."
        )

    # Define file paths
    # The script is in src/sekai_optimizer/scripts, so we go up three levels for the root
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "src" / "sekai_optimizer" / "data"

    stories_file = DATA_DIR / "stories.json"
    users_file = DATA_DIR / "users.json"

    # Initialize client and run synthesis
    openai_client = openai.OpenAI()
    synthesize_and_save_data(openai_client, stories_file, users_file)
    print("\nData synthesis complete.")
