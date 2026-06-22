import os
import shutil
import sys
import csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import anthropic
from config import ANTHROPIC_API_KEY, MODEL
import json
from config import VAULT_PATH

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

TEMPLATE_CODE = """Extract all technical content from this conversation for a data scientist's code notes.

Output clean markdown with these sections:

## Current State
What exists right now — function names, file names, key parameters, architecture.

For each function or component, include:
- Function signature
- Purpose: what it does in one sentence
- How it works: key implementation details
- Why: any non-obvious design decisions
- Location: file path

## Changes
Extract any formal change records from this conversation.
A change record must have ALL FOUR of these sections to qualify:
Proposal, Design, Tasks, Spec delta.
Partial discussions do not count.

For each complete change record found, output EXACTLY this format:
### CHANGE: [change-name-as-slug]
#### proposal
[proposal content]
#### design
[design content]
#### tasks
[tasks content]
#### spec_delta
[spec delta content]

Rules:
- Current State: function signatures, purpose, how it works, why, location ONLY
- Changes: only extract when all four sections are present — no partial changes
- Use lowercase hyphenated slugs for change names (e.g. other-investigator, not "Other Investigator")
- Skip test code, test corrections, and anything that only affects the test block
- Be specific and technical
- Include exact function names, file paths, parameter names
- If a section has nothing, omit it entirely

IMPORTANT: Only document code that is explicitly part of {project_name}.
Do not document purely hypothetical examples or code explicitly labeled as belonging to a different project.
If the conversation references a specific filename or says "create a file", "save this as", or "here's the complete [filename]", 
treat that code as production code for {project_name} and document it — even if introduced in a learning context.
If no production code for {project_name} exists in this conversation, write:
## Current State
Status: Design phase — no production code written yet.
Architecture decisions are captured in Planning/.

Project: {project_name}
Conversation chunk:
{conversation_chunk}"""

TEMPLATE_ERRORS_FIXES = """Extract all bugs, errors, and fixes from this conversation.

Output a numbered list only — no headers, no extra text.
Format each as:
1. WHAT: [one line describing the fix] | WHY: [one short sentence]

Rules:
- Only include actual bugs or errors that were found and fixed
- Skip test corrections, planned improvements, or future work
- Skip anything that only affects test code
- If nothing found, return exactly: NONE

Project: {project_name}
Conversation chunk:
{conversation_chunk}"""

TEMPLATE_PLANNING = """Extract planning and strategic content from this conversation.

Output clean markdown with these sections:

## What We're Building
A concise description of the project — what it is, what problem it solves,
and its key capabilities or components.

## Next Steps
Specific actionable items. Not "continue building" but exact next action.
Format each as:
- [ ] [specific action]

## Alternatives & Options
Other approaches discussed, options being considered.

## Context & Motivations
The WHY behind decisions. Why was this approach chosen over others?
Format each as:
DECISION: [what was decided]
REASON: [why]

## Open Questions
Unresolved questions, things that need investigation or decision.
Format each as:
- ? [question]

Rules:
- Focus on direction and reasoning, not implementation details
- Be specific — not "think about X" but "decide whether to use SqliteSaver or PostgresSaver"
- If a section has nothing, omit it entirely

Project: {project_name}
Conversation chunk:
{conversation_chunk}"""

TEMPLATE_EXPLORATION = """Extract exploration and research content from this conversation.

Output clean markdown with these sections:

## What We're Exploring
A concise description of the question or opportunity being investigated —
what it is, why it matters, and what a successful exploration would answer.

## What I've Found
Confirmed facts and findings from this exploration.
Format each confirmed finding as:
- ✓ [finding]

Anything still unclear or unverified:
- ? [unclear item]

## Decisions Made
What was decided during this exploration and why.
Format each as:
DECISION: [what was decided]
REASON: [why — what criteria or findings led to this]

## Blockers
What is outside your control or gating the next step.
Format each as:
- ⛔ [blocker — who or what is gating this]

## Next Steps
Specific actionable items. Not "continue exploring" but exact next action.
Format each as:
- [ ] [specific action]

## Open Questions
Unresolved questions that need investigation or a decision.
Format each as:
- ? [question]

## Transition Criteria
What would need to be true for this exploration to become a build project.
Format each as:
- [ ] [criterion]

Rules:
- Confirmed findings only go under What I've Found — not assumptions or hopes
- Decisions need both a what and a why — skip if reasoning isn't clear
- Be specific — not "follow up with vendor" but "reply to Deltek rep confirming REST/JSON vs Excel"
- If a section has nothing, omit it entirely

Project: {project_name}
Conversation chunk:
{conversation_chunk}"""

TEMPLATE_FILE_INDEX = """You are reading a file that belongs to an exploration project.
Write a single paragraph describing what this file is, what it contains,
and why it exists in the context of this project.

Be specific — not "this file contains information" but what the information
actually is and what decision or action it supports.
Write for someone who needs to decide quickly whether this file
is worth opening.

Project: {project_name}
File: {file_name}
Contents:
{file_content}"""

TEMPLATE_LEARNING = """Extract learning and knowledge content from this conversation.

Output clean markdown with these sections:

## Concepts Learned
New things understood for the first time or understood more deeply.
Be specific — not "learned about LangGraph" but "understood that conditional edges 
use a plain Python router function, not the LLM, to decide which node to go to next"

## Mental Models
New ways of thinking about something. Analogies, frameworks, principles.

## Instincts & Pushbacks
Personal observations, things questioned, gut feelings that proved right or wrong.
Format each as:
INSTINCT: [what you felt/thought]
OUTCOME: [what happened / was it right?]

## Connected Ideas
How this learning connects to other things you already know.

## Working Examples
Confirmed working code patterns worth referencing.
For each include:
- What it demonstrates
- Key code snippet or pattern
- Location
- Why it matters for future work

Rules:
- Use specific examples from the conversation
- Capture the reasoning, not just the conclusion
- Write as if explaining to future-you who needs to remember this
- If a section has nothing, omit it entirely

Project: {project_name}
Conversation chunk:
{conversation_chunk}"""

TEMPLATE_PROJECT_MAP = """Extract all files and resources mentioned in this conversation for a software engineer's project map.

Output clean markdown with these sections:

## Code Files
| File | Location | Purpose | Read when... |
|------|----------|---------|--------------|

## Notebooks
| File | Location | Purpose | Read when... |
|------|----------|---------|--------------|

## Data Files & Artifacts
| File | Schema/Structure | Purpose | Read when... |
|------|------------------|---------|--------------|

## Reference & Documentation
| File | Location | Purpose | Read when... |
|------|----------|---------|--------------|

## Auth & Config
| File | Location | Purpose | Read when... |
|------|----------|---------|--------------|

Rules:
- Only include files that are explicitly confirmed as existing in the conversation. Do not include planned or future files
- Only include files that belong to {project_name} 
- Do NOT include files from other projects even if mentioned in the conversation
- If a file belongs to a different project, skip it entirely
- Purpose: one short sentence, what the file does
- Read when: one short phrase, when an AI or developer should read this file
- For Data Files: include schema/structure details — column names, shape, file format, key fields
- If a section has no files, omit it entirely
- Be specific — not "utility file" but "contains extract_new_content(), read_file(), write_file(), save_anchor()"

Project: {project_name}
Conversation chunk:
{conversation_chunk}"""

TEMPLATE_CODE_INDEX = """You are reading a Python source file and writing a code index for a developer's reference.

Your output will be inserted into a larger code_index.md document. Write only the section for this file.

File: {file_name}
AST Skeleton (all functions/classes extracted):
{ast_skeleton}

Full source:
{source}

Output format:

## {file_name}
[2-3 sentence summary: what this file does, how it fits in the project, anything non-obvious about its structure]

For each function write:
### function_name(signature)
[2-3 sentences: what it does, how it works, any non-obvious design decisions worth knowing]

Rules:
- Cover every function in the AST skeleton — do not skip any
- Be specific about implementation details, not just purpose
- Note any important parameters, return values, or side effects
- If a function is a thin wrapper or stub, say so in one sentence
- Write for a developer who needs to orient quickly, not for documentation
- No preamble, no closing remarks — just the ## section and ### entries"""

TEMPLATE_TRANSITION_PLANNING = """You are seeding a planning document for a project that is graduating from exploration to active development.

Read the exploration note below and produce a planning document using this exact format:

## What We're Building
A concise description of what this project builds, what problem it solves, and what a successful Phase 0 looks like.
Draw from the "What We're Exploring" and "Transition Criteria" sections.

## Next Steps
Pull directly from the exploration "Next Steps" section.
These are already the technical backlog — preserve them as-is, formatted as:
- [ ] [specific action]

## Alternatives & Options
Pull from any alternatives, options, or trade-offs discussed in the exploration note.
If none were explicitly discussed, omit this section.

## Context & Motivations
Pull from the "Decisions Made" section of the exploration note.
Format each as:
DECISION: [what was decided]
REASON: [why]

## Open Questions
Pull from the "Open Questions" section of the exploration note.
If empty, omit this section entirely.

Rules:
- Do not invent content — only use what is in the exploration note
- Preserve specific details: API names, field names, cost figures, thresholds
- Output the planning document only — no preamble, no closing remarks

Project: {project_name}
Exploration note:
{exploration_content}"""


TEMPLATE_TRANSITION_STATE = """You are seeding an initial current_state.md for a project that is in design phase — no production code has been written yet.

Read the exploration note below and produce a current_state.md stub that captures what is already known about the architecture.

Output format:

## Current State
Status: Design phase — no production code written yet.
Architecture decisions and component responsibilities are captured below.

### Pipeline Overview
Describe the end-to-end pipeline in 3-5 sentences based on what was decided during exploration.
Include: data sources, processing stages, output mechanism, and scheduling approach.

### Components
For each distinct component identified during exploration, write:

#### [Component Name]
- **Responsibility:** what this component does
- **Key decisions:** any design decisions already made (API endpoints, auth approach, data format, etc.)
- **Input:** what it receives
- **Output:** what it produces
- **Notes:** anything non-obvious or deferred

### Data Schema
If a normalized schema was defined during exploration, document it here.
List each field with its name and purpose.

Rules:
- Only document what was confirmed during exploration — not speculation
- Preserve specific details: endpoint URLs, field names, rate limits, cost estimates
- If a component has no confirmed decisions yet, note it as "TBD"
- Output the document only — no preamble, no closing remarks
- Start directly with ## Current State

Project: {project_name}
Exploration note:
{exploration_content}"""

def extract_new_content(previous: str, new: str) -> str | None:
    import re
    
    # Option A: session marker takes priority
    markers = list(re.finditer(
        r'===\s*SESSION\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*===', 
        new
    ))
    
    if markers:
        last_marker = markers[-1]
        new_content = new[last_marker.end():].strip()
        print(f"✓ Session marker found — extracting content after {last_marker.group().strip()}")
        if not new_content:
            return ""
        return new_content
    
    # Option B: anchor-based diff fallback
    if not previous.strip():
        return new
    
    anchor = previous[-500:]
    position = new.rfind(anchor)
    
    if position == -1:
        return None
    
    new_content = new[position + len(anchor):]
    if not new_content.strip():
        return ""
    
    return new_content.strip()

def read_file(path: str) -> str:
    """
    Read a file and return its contents as a string.
    Returns "" if the file doesn't exist.
    """
    from pathlib import Path
    
    file = Path(path)
    
    if not file.exists():
        return ""
    
    return file.read_text(encoding="utf-8")

def write_file(path: str, content: str, mode: str = "append") -> None:
    """
    Write content to a file.
    Creates parent folders if they don't exist.
    
    mode: "append" — adds to bottom (default)
          "overwrite" — replaces everything
    """
    from pathlib import Path
    
    file = Path(path)
    
    # create parent folders if they don't exist
    file.parent.mkdir(parents=True, exist_ok=True)
    
    if mode == "append":
        with open(file, "a", encoding="utf-8") as f:
            f.write("\n\n" + content)
    elif mode == "overwrite":
        with open(file, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Use 'append' or 'overwrite'.")
    

def save_anchor(conversation: str, anchor_path: str) -> None:
    """
    Save the last 500 characters of a conversation as an anchor.
    Used next time to find where we left off.
    """
    anchor = conversation[-500:]
    write_file(anchor_path, anchor, mode="overwrite")

def snapshot_project(project_name: str) -> None:
    """
    Save a snapshot of the current project notes before overwriting.
    Copies Planning/, Learning/, and Code/ (including changes/) into
    <vault>/<project_folder>/archive/, always replacing the previous snapshot.
    Called once per project at the top of Step 7, before any writes.
    Skip for Personal project.
    """
    if project_name == "Personal":
        return

    vault = Path(VAULT_PATH)
    folder_name = project_name.replace(" ", "_")
    project_path = vault / folder_name
    archive_path = project_path / "archive"

    folders_to_copy = ["Planning", "Learning", "Code"]

    # wipe existing archive
    if archive_path.exists():
        shutil.rmtree(archive_path)
    archive_path.mkdir(parents=True, exist_ok=True)

    copied = []
    for folder in folders_to_copy:
        src = project_path / folder
        if src.exists():
            shutil.copytree(src, archive_path / folder)
            copied.append(folder)
    
    # ← ADD THIS BLOCK
    exploration_file = project_path / f"{folder_name}_Exploration.md"
    if exploration_file.exists():
        shutil.copy2(exploration_file, archive_path / f"{folder_name}_Exploration.md")
        copied.append(f"{folder_name}_Exploration.md")

    if copied:
        print(f"  ✓ Snapshot saved for {project_name}: {', '.join(copied)}")
    else:
        print(f"  ↷ Nothing to snapshot for {project_name} — folders not found")


def log_usage(project_name: str, function_name: str, response) -> dict:
    """
    Extract token usage from an API response and calculate cost.
    Returns a usage dict for accumulation in process_conversation.
    """
    from config import COST_PER_INPUT_TOKEN, COST_PER_OUTPUT_TOKEN

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)

    return {
        "project": project_name,
        "function": function_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
    }

def append_usage_log(rows: list[dict]) -> None:
    """
    Append per-call usage rows plus a TOTAL row to usage_log.csv.
    Creates the file with headers if it doesn't exist.
    """
    from datetime import datetime
    from config import USAGE_LOG_PATH

    log_path = Path(USAGE_LOG_PATH)
    write_header = not log_path.exists()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    total_input = sum(r["input_tokens"] for r in rows)
    total_output = sum(r["output_tokens"] for r in rows)
    total_cost = sum(r["cost_usd"] for r in rows)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "project", "function", "input_tokens", "output_tokens", "cost_usd"]
        )
        if write_header:
            writer.writeheader()

        for row in rows:
            writer.writerow({
                "timestamp": timestamp,
                "project": row["project"],
                "function": row["function"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cost_usd": row["cost_usd"],
            })

        # TOTAL row
        writer.writerow({
            "timestamp": timestamp,
            "project": "TOTAL",
            "function": "-",
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cost_usd": round(total_cost, 6),
        })

    print(f"\n✓ Run cost: ${total_cost:.4f} (input: {total_input:,}, output: {total_output:,})")
    print(f"  Usage logged to: {USAGE_LOG_PATH}")

def detect_projects(conversation_chunk: str, usage_rows: list) -> list:
    # truncate to last 50K chars if too long
    MAX_CHUNK = 150000
    if len(conversation_chunk) > MAX_CHUNK:
        print(f"⚠ Chunk too large ({len(conversation_chunk)} chars) — truncating to last {MAX_CHUNK}")
        conversation_chunk = conversation_chunk[-MAX_CHUNK:]
    """
    Ask Claude to identify all projects/topics discussed in the conversation.
    Returns a list of dicts with project, name, type, and description.
    """
    prompt = f"""Read this conversation and identify all distinct topics discussed.
Group related topics under the same project name.

For each topic return a JSON array with objects containing:
- "project": the project this topic belongs to (e.g. "Personal Memory Agent", "LangGraph QC Agent")
- "name": short topic name
- "type": one of TECHNICAL, LEARNING, PERSONAL, EXPLORATION
- "description": one sentence describing what was discussed

Rules:
- TECHNICAL: code, architecture, design decisions, bugs, fixes
- LEARNING: only extract concepts that are genuinely new and non-obvious.
  Skip general programming principles unless they were a specific revelation.
  Combine related learning concepts into one entry rather than splitting them.
- EXPLORATION: research, vendor evaluation, feasibility investigation, decision-making before any code exists.
- PERSONAL: career, emotions, non-technical conversations
- Group topics that belong to the same project under the same "project" value
- Use consistent project names across all topics
- If a learning topic is relevant to multiple projects, assign project as "Personal"
- PERSONAL type topics should always use project "Personal"

Return ONLY a valid JSON array, no preamble, no markdown backticks.

Example output:
[
    {{"project": "Personal Memory Agent", "name": "extract_new_content function", "type": "TECHNICAL", "description": "Built and tested the diff function"}},
    {{"project": "Personal Memory Agent", "name": "LangChain Fundamentals", "type": "LEARNING", "description": "Understood tool calling and agent loop"}},
    {{"project": "LangGraph QC Agent", "name": "QC Agent Architecture", "type": "TECHNICAL", "description": "Designed four-capability QC layer"}}
]

Conversation:
{conversation_chunk}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )

    usage_rows.append(log_usage("pipeline", "detect_projects", response))

    raw = response.content[0].text.strip()

    # strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]).strip()

    try:
        projects = json.loads(raw)
        return projects
    except json.JSONDecodeError:
        print(f"⚠ Could not parse response as JSON:\n{raw}")
        return []

def human_checkpoint(projects: list, allow_reassign: bool = False) -> tuple:
    """
    Displays detected topics and returns (selected, reassigned) tuple.
    
    selected: list of topics to process for current project
    reassigned: dict of {project_name: [topics]} to route elsewhere
    """
    if not projects:
        print("No topics to review.")
        return [], {}

    print("\nDetected projects/topics:\n")
    for i, project in enumerate(projects, start=1):
        print(f"  {i}. [{project['type']}] {project['name']} — {project['description']}")

    print("\nWhich would you like to keep?")
    if allow_reassign:
        print("Options:")
        print("  - Numbers to keep (e.g. 1,2,3)")
        print("  - Prefix with project to reassign (e.g. Personal:1,2)")
        print("  - Prefix with type to retype (e.g. EXPLORATION:1,2)")
        print("  - Combine both (e.g. 1,2,3 Personal:7,8 EXPLORATION:3)")
        print("  - 'all' or 'none'")
    else:
        print("Enter numbers separated by commas (e.g. 1,2) or 'all' or 'none':")

    user_input = input("> ").strip()

    selected = []
    reassigned = {}

    if user_input.lower() == "all":
        return projects, {}
    elif user_input.lower() == "none":
        return [], {}

    # parse input — split by spaces to find segments
    segments = user_input.split()

    keep_indices = []
    VALID_TYPES = {"TECHNICAL", "LEARNING", "PERSONAL", "EXPLORATION"}

    for segment in segments:
        if ":" in segment:
            parts = segment.split(":", 1)
            prefix = parts[0].strip().upper()
            try:
                indices = [int(x.strip()) - 1 for x in parts[1].split(",")]
                if prefix in VALID_TYPES:
                    # type override
                    for i in indices:
                        if 0 <= i < len(projects):
                            projects[i]["type"] = prefix
                            print(f"  ✓ Retyped topic {i+1} → {prefix}")
                else:
                    # project reassignment
                    target_project = parts[0].replace("_", " ")
                    topics = [projects[i] for i in indices if 0 <= i < len(projects)]
                    if target_project not in reassigned:
                        reassigned[target_project] = []
                    reassigned[target_project].extend(topics)
            except (ValueError, IndexError):
                print(f"  ⚠ Could not parse: {segment}")
        else:
            # regular keep
            try:
                indices = [int(x.strip()) - 1 for x in segment.split(",")]
                keep_indices.extend(indices)
            except ValueError:
                print(f"  ⚠ Could not parse: {segment}")

    selected = [projects[i] for i in keep_indices if 0 <= i < len(projects)]

    print(f"\n✓ Keeping {len(selected)} topics")
    if reassigned:
        for proj, topics in reassigned.items():
            print(f"  → Reassigning {len(topics)} topic(s) to: {proj}")

    return selected, reassigned

def parse_code_output(raw: str) -> dict:
    """
    Parse TEMPLATE_CODE output into current_state and changes.
    
    Returns dict with:
        "current_state": str — everything before ## Changes
        "changes": list of dicts — [{name, proposal, design, tasks, spec_delta}]
    """
    result = {"current_state": "", "changes": []}
    
    # Split on ## Changes
    if "## Changes" in raw:
        parts = raw.split("## Changes", 1)
        result["current_state"] = parts[0].strip()
        changes_raw = parts[1].strip()
    else:
        result["current_state"] = raw.strip()
        return result
    
    # Parse individual changes — split on ### CHANGE:
    import re
    change_blocks = re.split(r'###\s+CHANGE:\s+', changes_raw)
    
    for block in change_blocks:
        if not block.strip():
            continue
        
        lines = block.strip().split("\n")
        change_name = lines[0].strip()
        block_content = "\n".join(lines[1:])
        
        # Extract four sections
        sections = {}
        for section in ["proposal", "design", "tasks", "spec_delta"]:
            pattern = rf'####\s+{section}\s*\n(.*?)(?=####|\Z)'
            match = re.search(pattern, block_content, re.IGNORECASE | re.DOTALL)
            sections[section] = match.group(1).strip() if match else ""
        
        # Only add if all four sections present
        if all(sections.values()):
            sections["name"] = change_name
            result["changes"].append(sections)
        else:
            print(f"  ⚠ Incomplete change record '{change_name}' — skipping")
    
    return result

def summarize_for_notes(conversation_chunk: str, project: dict, usage_rows: list) -> dict:
    """
    Extract structured notes from a conversation chunk for a given topic.
    
    Note: current_state, project_map, changes, planning, and learning
    are all handled at project level in Step 7 of process_conversation().
    This function only handles per-topic errors/fixes checkpoint.
    
    Returns a dict with keys:
    {
        "errors_fixes": "numbered list",   # if TECHNICAL and ENABLE_ERRORS_CHECKPOINT
    }
    """
    results = {}
    project_name = project["name"]
    project_type = project["type"]

    if project_type == "TECHNICAL":
        # errors & fixes (only if checkpoint enabled)
        from config import ENABLE_ERRORS_CHECKPOINT
        if ENABLE_ERRORS_CHECKPOINT:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1000,
                messages=[{"role": "user", "content": TEMPLATE_ERRORS_FIXES.format(
                    project_name=project_name,
                    conversation_chunk=conversation_chunk
                )}]
            )
            usage_rows.append(log_usage(project_name, "summarize_for_notes", response))
            results["errors_fixes"] = response.content[0].text.strip()

    elif project_type == "LEARNING":
        # learning handled at project level in process_conversation()
        pass
    
    elif project_type == "EXPLORATION":
        pass

    elif project_type == "PERSONAL":
        print(f"  ↷ Skipping PERSONAL topic: {project_name}")

    return results

def summarize_planning(conversation_chunk: str, project_name: str, existing_planning: str, usage_rows: list) -> str:
    """
    Generate or update a planning note for a project.
    If existing_planning is provided, merges intelligently.
    Called once per project per run.
    """
    if existing_planning.strip():
        prompt = f"""You are updating a planning document for {project_name}.

Existing planning document:
{existing_planning}

New conversation to extract planning from:
{conversation_chunk}

Produce an updated planning document that:
- Always preserve and update the "What We're Building" section
- This is the first thing a new session should read
- Merges new next steps with existing ones
- Removes next steps that are clearly completed based on new conversation
- Updates decisions section with new decisions
- Adds new open questions, removes resolved ones
- Keeps alternatives section current


Output the final planning document only. 
Do not include any preamble, reasoning, or notes about what you changed."""
    else:
        prompt = TEMPLATE_PLANNING.format(
            project_name=project_name,
            conversation_chunk=conversation_chunk
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )
    usage_rows.append(log_usage(project_name, "summarize_planning", response))
    return response.content[0].text.strip()

def summarize_exploration(conversation_chunk: str, project_name: str, existing_notes: str, usage_rows: list) -> str:
    """
    Generate or update an exploration note for a project.
    If existing_notes is provided, merges intelligently.
    Called once per project per run.
    """
    if existing_notes.strip():
        prompt = f"""You are updating an exploration document for {project_name}.

Existing exploration document:
{existing_notes}

New conversation to extract exploration content from:
{conversation_chunk}

Produce an updated exploration document that:
- Always preserve and update the "What We're Exploring" section
- Adds new confirmed findings, marks previously unclear items as confirmed if resolved
- Removes open questions that have been answered
- Updates blockers — remove any that are resolved, add new ones
- Merges new decisions with existing ones, no duplicates
- Updates next steps — remove completed items, add new ones
- Updates transition criteria if new conditions were identified

Output the final exploration document only.
Do not include any preamble, reasoning, or notes about what you changed."""
    else:
        prompt = TEMPLATE_EXPLORATION.format(
            project_name=project_name,
            conversation_chunk=conversation_chunk
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )
    usage_rows.append(log_usage(project_name, "summarize_exploration", response))
    return response.content[0].text.strip()

def summarize_learning(conversation_chunk: str, project_name: str, existing_learning: str, usage_rows: list) -> str:
    """
    Generate or update a learning note for a project.
    If existing_learning is provided, merges intelligently.
    Called once per project per run.
    """
    if existing_learning.strip():
        prompt = f"""You are updating a learning document for {project_name}.

Existing learning document:
{existing_learning}

New conversation to extract learning from:
{conversation_chunk}

Produce an updated learning document that:
- Merges new concepts with existing ones — no duplicate entries
- Expands existing concepts if new detail was added
- Adds new mental models, instincts, and connected ideas
- Removes redundant or repeated explanations
- Keeps the best version of each concept, not all versions

Output the full updated learning document only, no preamble."""

    else:
        prompt = TEMPLATE_LEARNING.format(
            project_name=project_name,
            conversation_chunk=conversation_chunk
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )
    usage_rows.append(log_usage(project_name, "summarize_learning", response))
    return response.content[0].text.strip()

def summarize_current_state(conversation_chunk: str, project_name: str, existing_state: str, usage_rows: list) -> str:
    """
    Generate or merge current_state.md for a project.
    Called once per project per run.
    """
    if existing_state.strip():
        prompt = f"""You are updating a code state document for {project_name}.

Existing current_state.md:
{existing_state}

New conversation to extract code state from:
{conversation_chunk}

Produce an updated current_state.md that:
- Adds new functions with full signature, purpose, how it works, why, location
- Updates existing functions if their implementation changed
- Preserves all existing functions not mentioned in the new conversation unchanged
- Only removes a function if the conversation explicitly states it was deleted or removed

Output the final document only. No preamble, no reasoning, no notes about what changed.
Start directly with ## Current State"""
    else:
        prompt = TEMPLATE_CODE.format(
            project_name=project_name,
            conversation_chunk=conversation_chunk
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )
    usage_rows.append(log_usage(project_name, "summarize_current_state", response))
    content = response.content[0].text.strip()
    if "##" in content:
        content = content[content.index("##"):].strip()
    return content

def summarize_project_map(conversation_chunk: str, project_name: str, existing_map: str, usage_rows: list) -> str:
    """
    Generate or merge project_map.md for a project.
    Called once per project per run.
    """
    if existing_map.strip():
        prompt = f"""You are updating a project map for {project_name}.

Existing project_map.md:
{existing_map}

New conversation to extract file references from:
{conversation_chunk}

Produce an updated project_map.md that:
- Only includes files belonging to {project_name}
- Removes any files that belong to other projects
- Adds any new files confirmed as existing in the new conversation
- Updates descriptions if a file's purpose changed
- Preserves all existing entries not mentioned in the new conversation
- Only removes a file if the conversation explicitly states it was deleted

Output the final document only. No preamble.
Only include files explicitly confirmed as existing — not planned or future files."""
    else:
        prompt = TEMPLATE_PROJECT_MAP.format(
            project_name=project_name,
            conversation_chunk=conversation_chunk
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}]
    )
    usage_rows.append(log_usage(project_name, "summarize_project_map", response))
    content = response.content[0].text.strip()
    if "##" in content:
        content = content[content.index("##"):].strip()
    return content

def errors_checkpoint(errors_fixes_raw: str) -> list:
    """
    Show detected errors & fixes to the user and ask which ones to document.
    Returns only the selected items as a list of strings.
    """
    if not errors_fixes_raw.strip() or errors_fixes_raw.strip() == "NONE":
        print("No errors or fixes detected.")
        return []

    # parse numbered list into individual items
    lines = [
        line.strip() for line in errors_fixes_raw.split("\n")
        if line.strip() and line.strip()[0].isdigit()
    ]

    if not lines:
        print("No errors or fixes detected.")
        return []

    print("\nErrors & fixes detected:\n")
    for i, line in enumerate(lines, start=1):
        print(f"  {i}. {line}")

    print("\nWhich are worth documenting in known_errors_fixes.md?")
    print("Enter numbers separated by commas (e.g. 1,2) or 'all' or 'none':")

    user_input = input("> ").strip().lower()

    if user_input == "all":
        selected = lines
    elif user_input == "none":
        selected = []
    else:
        try:
            indices = [int(x.strip()) - 1 for x in user_input.split(",")]
            selected = [lines[i] for i in indices if 0 <= i < len(lines)]
        except ValueError:
            print("⚠ Invalid input — keeping all items")
            selected = lines

    print(f"\n✓ Documenting {len(selected)} error(s)/fix(es)")
    return selected


def write_notes(results: dict, project_name: str) -> None:
    """
    Write summarized notes to the correct folders in the vault.
    
    results: output from summarize_for_notes() + approved errors/fixes
    project_name: used to build the folder path
    """
    from config import VAULT_PATH

    folder_name = project_name.replace(" ", "_")
    project_path = Path(VAULT_PATH) / folder_name

    # current state and project map — skip for Personal project
    # if project_name != "Personal":
        # if "current_state" in results:
        #     path = project_path / "Code" / "current_state.md"
        #     write_file(str(path), results["current_state"], mode="overwrite")
        #     print(f"  ✓ current_state.md updated")

        # if "project_map" in results:
        #     path = project_path / "Code" / "project_map.md"
        #     write_file(str(path), results["project_map"], mode="overwrite")
        #     print(f"  ✓ project_map.md updated")

    # errors & fixes — append approved items only
    if "approved_errors" in results and results["approved_errors"]:
        path = project_path / "Code" / "known_errors_fixes.md"
        content = "\n".join([f"- {item}" for item in results["approved_errors"]])
        write_file(str(path), content, mode="append")
        print(f"  ✓ known_errors_fixes.md updated")

    # # planning — append
    # if "planning" in results:
    #     path = project_path / "Planning" / f"{folder_name}_Planning.md"
    #     write_file(str(path), results["planning"], mode="append")
    #     print(f"  ✓ Planning updated")

def generate_bookmark(inbox_file: str) -> str:
    """
    Generate a session timestamp marker, append to inbox file,
    copy to clipboard, show popup.
    """
    from datetime import datetime
    import tkinter as tk
    from tkinter import messagebox

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    marker = f"=== SESSION {timestamp} ==="

    write_file(inbox_file, f"\n{marker}", mode="append")

    root = tk.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(marker)
    root.update()

    messagebox.showinfo(
        "Session Bookmark Ready",
        f"Paste this into Claude to bookmark your session:\n\n{marker}"
    )
    root.destroy()

    print(f"✓ Session bookmark: {marker}")
    return marker

def extract_changes(conversation_chunk: str, project_name: str, usage_rows: list) -> list:
    """
    Extract formal change records from a conversation chunk for a project.
    Called once per project per run.
    Returns list of change dicts: [{name, proposal, design, tasks, spec_delta}]
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": TEMPLATE_CODE.format(
            project_name=project_name,
            conversation_chunk=conversation_chunk
        )}]
    )
    usage_rows.append(log_usage(project_name, "extract_changes", response))
    parsed = parse_code_output(response.content[0].text.strip())
    
    # deduplicate — keep last occurrence of each change name
    seen = {}
    for change in parsed["changes"]:        # ← fix 1: parsed["changes"]
        seen[change["name"]] = change
    changes = list(seen.values())
    
    if changes:
        print(f"  ✓ Found {len(changes)} change record(s) for {project_name}")
    
    return changes                           # ← fix 2: return deduplicated list

def extract_ast_skeleton(source: str) -> str:
    """
    Use ast to extract all top-level function and class definitions
    with their signatures and docstrings.
    Returns a compact string representation for use in LLM prompts.
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"⚠ Could not parse file: {e}"

    lines = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = []
            for arg in node.args.args:
                annotation = ""
                if arg.annotation:
                    annotation = f": {ast.unparse(arg.annotation)}"
                args.append(f"{arg.arg}{annotation}")

            returns = ""
            if node.returns:
                returns = f" -> {ast.unparse(node.returns)}"

            signature = f"def {node.name}({', '.join(args)}){returns}"
            docstring = ast.get_docstring(node)
            if docstring:
                first_line = docstring.strip().split("\n")[0]
                lines.append(f"{signature}\n    # {first_line}")
            else:
                lines.append(signature)

        elif isinstance(node, ast.ClassDef):
            lines.append(f"class {node.name}:")
            docstring = ast.get_docstring(node)
            if docstring:
                first_line = docstring.strip().split("\n")[0]
                lines.append(f"    # {first_line}")

    return "\n".join(lines)

def generate_code_tree(project_name: str, repo_summary: str, file_data: list) -> None:
    """
    Generates a human-readable code_tree.txt with full repo and file summaries
    as paragraphs, followed by a function tree with summaries truncated to 100 chars.
    Called at the end of summarize_code_index.
    """
    vault = Path(VAULT_PATH)
    folder_name = project_name.replace(" ", "_")
    output_path = vault / folder_name / "Code" / "code_tree.txt"

    divider = "─" * 60
    lines = []

    # repo summary
    lines.append(f"# {project_name}")
    lines.append("")
    lines.append(repo_summary)
    lines.append("")

    for i, file in enumerate(file_data):
        lines.append(divider)
        lines.append("")

        # file name + summary paragraph
        lines.append(file["file_name"])
        if file["file_summary"]:
            lines.append(file["file_summary"])
        lines.append("")

        # function tree
        functions = file["functions"]
        if functions:
            for j, func in enumerate(functions):
                prefix = "└──" if j == len(functions) - 1 else "├──"
                summary = func["summary"]
                lines.append(f"{prefix} {func['name']} — {summary}")
        else:
            lines.append("└── [constants only]")

        lines.append("")

    final_output = "\n".join(lines)
    write_file(str(output_path), final_output, mode="overwrite")
    print(f"✓ code_tree.txt written to: {output_path}")

def summarize_code_index(project_name: str) -> None:
    """
    Generate code_index.md for a project by reading source files from disk,
    extracting ast skeletons, and calling the LLM once per file (or per chunk
    if file exceeds 120K chars). Writes a tree-structured markdown document to
    <vault>/<project_folder>/Code/code_index.md.
    Called on-demand, not wired into process_conversation.
    """
    import ast
    from config import PROJECT_SOURCE_FILES

    file_paths = PROJECT_SOURCE_FILES.get(project_name, [])
    if not file_paths:
        print(f"⚠ No source files configured for '{project_name}' in PROJECT_SOURCE_FILES")
        return

    vault = Path(VAULT_PATH)
    folder_name = project_name.replace(" ", "_")
    output_path = vault / folder_name / "Code" / "code_index.md"

    usage_rows = []
    file_sections = []
    file_data = []
    repo_summary = ""

    print(f"\n{'='*60}")
    print(f"Building code index for: {project_name}")
    print(f"{'='*60}")

    for i, file_path in enumerate(file_paths):
        path = Path(file_path)
        if not path.exists():
            print(f"  ⚠ File not found: {file_path} — skipping")
            continue

        file_name = path.name
        source = path.read_text(encoding="utf-8")
        skeleton = extract_ast_skeleton(source)

        print(f"\n  Processing: {file_name} ({len(source):,} chars)")

        CHUNK_THRESHOLD = 120000

        if len(source) <= CHUNK_THRESHOLD:
            # single pass
            prompt = TEMPLATE_CODE_INDEX.format(
                file_name=file_name,
                ast_skeleton=skeleton,
                source=source
            )
            response = client.messages.create(
                model=MODEL,
                max_tokens=18000,
                messages=[{"role": "user", "content": prompt}]
            )
            usage_rows.append(log_usage(project_name, f"summarize_code_index:{file_name}", response))
            section = response.content[0].text.strip()

        else:
            # chunked pass — split on top-level ast boundaries
            print(f"  ⚠ File exceeds {CHUNK_THRESHOLD:,} chars — chunking by ast boundaries")
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                print(f"  ⚠ Could not parse {file_name}: {e} — skipping")
                continue

            source_lines = source.splitlines(keepends=True)
            top_level = [
                node for node in ast.iter_child_nodes(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]

            chunks = []
            current_chunk_lines = []
            current_chunk_size = 0

            for node in top_level:
                start = node.lineno - 1
                end = node.end_lineno
                node_lines = source_lines[start:end]
                node_size = sum(len(l) for l in node_lines)

                if current_chunk_size + node_size > CHUNK_THRESHOLD and current_chunk_lines:
                    chunks.append("".join(current_chunk_lines))
                    current_chunk_lines = node_lines
                    current_chunk_size = node_size
                else:
                    current_chunk_lines.extend(node_lines)
                    current_chunk_size += node_size

            if current_chunk_lines:
                chunks.append("".join(current_chunk_lines))

            chunk_sections = []
            for j, chunk in enumerate(chunks):
                chunk_skeleton = extract_ast_skeleton(chunk)
                prompt = TEMPLATE_CODE_INDEX.format(
                    file_name=f"{file_name} (part {j+1}/{len(chunks)})",
                    ast_skeleton=chunk_skeleton,
                    source=chunk
                )
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=18000,
                    messages=[{"role": "user", "content": prompt}]
                )
                usage_rows.append(log_usage(project_name, f"summarize_code_index:{file_name}:chunk{j+1}", response))
                chunk_sections.append(response.content[0].text.strip())

            section = "\n\n".join(chunk_sections)

        # first file also generates repo summary
        if i == 0:
            repo_prompt = f"""In 3-4 sentences, describe the overall purpose and architecture of the {project_name} codebase.
What problem does it solve? What are its major components or pipeline stages?
Write for a developer orienting themselves for the first time.
No preamble — start directly with the description.

Main file source:
{source[:8000]}"""

            repo_response = client.messages.create(
                model=MODEL,
                max_tokens=15000,
                messages=[{"role": "user", "content": repo_prompt}]
            )
            usage_rows.append(log_usage(project_name, "summarize_code_index:repo_summary", repo_response))
            repo_summary = repo_response.content[0].text.strip()

        # parse file summary and function summaries for code_tree
        file_summary = ""
        functions = []

        lines = section.split("\n")
        # file summary = lines after ## header until first ### or blank cluster
        in_file_summary = False
        for line in lines:
            if line.startswith("## "):
                in_file_summary = True
                continue
            if line.startswith("### "):
                break
            if in_file_summary and line.strip():
                file_summary += line.strip() + " "

        # function summaries = first sentence after each ### header
        current_func = None
        for line in lines:
            if line.startswith("### "):
                current_func = line[4:].strip()
                # strip signature — keep just the name
                current_func = current_func.split("(")[0].strip()
            elif current_func and line.strip():
                summary = line.strip()
                functions.append({"name": current_func, "summary": summary})
                current_func = None

        file_data.append({
            "file_name": file_name,
            "file_summary": file_summary.strip(),
            "functions": functions
        })

        file_sections.append(section)
        print(f"  ✓ {file_name} indexed")

    if not file_sections:
        print("⚠ No files were indexed — code_index.md not written")
        return

    # assemble final document
    output_parts = [
        f"# Code Index — {project_name}\n",
        f"## Repository\n{repo_summary}\n",
    ] + file_sections

    final_output = "\n\n".join(output_parts)
    write_file(str(output_path), final_output, mode="overwrite")

    generate_code_tree(project_name, repo_summary, file_data)

    append_usage_log(usage_rows)

    print(f"\n✓ code_index.md written to: {output_path}")

def summarize_file_index(project_name: str, folder_path: str) -> None:
    """
    Generate a file_index.md for an exploration project folder.
    Reads all files in folder_path, calls LLM once per file for a
    one-paragraph description, writes output to Planning/file_index.md.
    Called on-demand, not wired into process_conversation.
    """
    import pathlib

    folder = pathlib.Path(folder_path)
    if not folder.exists():
        print(f"⚠ Folder not found: {folder_path}")
        return

    vault = Path(VAULT_PATH)
    folder_name = project_name.replace(" ", "_")
    output_path = vault / folder_name / "file_index.md"

    usage_rows = []
    sections = []

    print(f"\n{'='*60}")
    print(f"Building file index for: {project_name}")
    print(f"Folder: {folder_path}")
    print(f"{'='*60}")

    files = [f for f in folder.iterdir() if f.is_file()]
    if not files:
        print("⚠ No files found in folder.")
        return

    for file_path in files:
        print(f"\n  Processing: {file_path.name}")

        # read content based on file type
        ext = file_path.suffix.lower()
        content = ""

        try:
            if ext in [".md", ".txt", ".csv", ".json"]:
                content = file_path.read_text(encoding="utf-8")

            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(str(file_path))
                    content = "\n".join(
                        page.extract_text() for page in reader.pages
                        if page.extract_text()
                    )
                except ImportError:
                    print(f"  ⚠ pypdf not installed — skipping {file_path.name}")
                    continue

            elif ext == ".docx":
                try:
                    import docx
                    doc = docx.Document(str(file_path))
                    content = "\n".join(p.text for p in doc.paragraphs if p.text)
                except ImportError:
                    print(f"  ⚠ python-docx not installed — skipping {file_path.name}")
                    continue

            else:
                print(f"  ↷ Unsupported file type: {ext} — skipping")
                continue

        except Exception as e:
            print(f"  ⚠ Could not read {file_path.name}: {e} — skipping")
            continue

        if not content.strip():
            print(f"  ↷ Empty file — skipping")
            continue

        # truncate to control cost
        MAX_CONTENT = 32000
        if len(content) > MAX_CONTENT:
            content = content[:MAX_CONTENT]
            print(f"  ⚠ Truncated to {MAX_CONTENT} chars")

        # call LLM
        prompt = TEMPLATE_FILE_INDEX.format(
            project_name=project_name,
            file_name=file_path.name,
            file_content=content
        )
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        usage_rows.append(log_usage(project_name, f"summarize_file_index:{file_path.name}", response))
        summary = response.content[0].text.strip()
        sections.append(f"### {file_path.name}\n{summary}")
        print(f"  ✓ Summarized: {file_path.name}")

    if not sections:
        print("⚠ No files could be summarized.")
        return

    output = "## File Index\n\n" + "\n\n".join(sections)
    write_file(str(output_path), output, mode="overwrite")
    print(f"\n✓ file_index.md written to: {output_path}")

    if usage_rows:
        append_usage_log(usage_rows)

def transition_to_technical(project_name: str) -> None:
    """
    Graduates an EXPLORATION project to a full technical project.
    Reads the completed exploration note and seeds planning.md and
    current_state.md stubs via two LLM calls.
    Creates Planning/, Learning/, and Code/changes/ folders.
    Exploration note is left untouched at project root.
    Call once manually from __main__ when exploration is complete.
    """
    vault = Path(VAULT_PATH)
    folder_name = project_name.replace(" ", "_")
    project_path = vault / folder_name

    # --- guard: project folder must exist ---
    if not project_path.exists():
        print(f"✗ Project folder not found: {project_path}")
        return

    # --- guard: exploration note must exist ---
    exploration_path = project_path / f"{folder_name}_Exploration.md"
    if not exploration_path.exists():
        print(f"✗ Exploration note not found: {exploration_path}")
        return

    exploration_content = read_file(str(exploration_path))
    print(f"\n{'='*60}")
    print(f"Transitioning to technical: {project_name}")
    print(f"{'='*60}")

    usage_rows = []

    # --- create technical folder structure ---
    folders = [
        project_path / "Planning",
        project_path / "Learning",
        project_path / "Code" / "changes",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    print("✓ Folder structure created: Planning/, Learning/, Code/changes/")

    # --- LLM call 1: seed planning.md ---
    print("\nGenerating planning.md...")
    planning_response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": TEMPLATE_TRANSITION_PLANNING.format(
            project_name=project_name,
            exploration_content=exploration_content
        )}]
    )
    usage_rows.append(log_usage(project_name, "transition_planning", planning_response))
    planning_content = planning_response.content[0].text.strip()
    planning_path = project_path / "Planning" / f"{folder_name}_Planning.md"
    write_file(str(planning_path), planning_content, mode="overwrite")
    print(f"✓ planning.md written → Planning/{folder_name}_Planning.md")

    # --- LLM call 2: seed current_state.md ---
    print("\nGenerating current_state.md stub...")
    state_response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        messages=[{"role": "user", "content": TEMPLATE_TRANSITION_STATE.format(
            project_name=project_name,
            exploration_content=exploration_content
        )}]
    )
    usage_rows.append(log_usage(project_name, "transition_state", state_response))
    state_content = state_response.content[0].text.strip()
    if "##" in state_content:
        state_content = state_content[state_content.index("##"):].strip()
    state_path = project_path / "Code" / "current_state.md"
    write_file(str(state_path), state_content, mode="overwrite")
    print(f"✓ current_state.md written → Code/current_state.md")

    print(f"\n{'='*60}")
    print(f"✓ Transition complete for {project_name}")
    print(f"  Exploration note preserved at: {folder_name}_Exploration.md")
    print(f"{'='*60}")

    append_usage_log(usage_rows)

def process_conversation(inbox_file: str, project_folder: str) -> None:
    """
    Full pipeline: read a conversation file from inbox, extract what's new,
    detect projects, set up folders, summarize and write notes per project.
    """
    from config import VAULT_PATH

    print(f"\n{'='*60}")
    print(f"Processing: {inbox_file}")
    print(f"{'='*60}")

    # usage accumulator — passed into every LLM-calling function
    usage_rows = []

    # Step 1: read new conversation
    new_conversation = read_file(inbox_file)
    if not new_conversation.strip():
        print("⚠ File is empty — nothing to process")
        return

    # Step 2: read previous conversation (anchor)
    anchor_path = str(Path(project_folder) / "inbox" / "anchor.txt")
    previous_conversation = read_file(anchor_path)

    # Step 3: extract new content
    new_chunk = extract_new_content(previous_conversation, new_conversation)

    if new_chunk is None:
        print("⚠ Anchor not found — treating full conversation as new content")
        new_chunk = new_conversation
        generate_bookmark(inbox_file)
    elif new_chunk == "":
        print("✓ No new content — notes are already up to date")
        return
    else:
        print(f"✓ Found {len(new_chunk)} characters of new content")
        generate_bookmark(inbox_file)

    # Step 4: detect projects
    print("\nDetecting projects/topics...")
    topics = detect_projects(new_chunk, usage_rows)

    if not topics:
        print("⚠ No topics detected — nothing to summarize")
        return

   # Step 5: group and rename projects
    grouped = group_and_rename_projects(topics)

    if not grouped:
        print("✓ Nothing to process — done")
        return

   # Step 6: for each project, human checkpoint then summarize
    processed_projects = set()  # track which projects had topics selected

    # Pass 1: run human_checkpoint for all projects to collect type corrections
    selected_per_project = {}
    for project_name, project_topics in list(grouped.items()):

        # skip Personal type topics entirely
        non_personal = [t for t in project_topics if t["type"] != "PERSONAL"]
        if not non_personal:
            print(f"\n  ↷ Skipping {project_name} — only PERSONAL topics")
            continue

        print(f"\n{'─'*60}")
        print(f"Project: {project_name}")
        print(f"{'─'*60}")

        print(f"\nTopics for {project_name}:")
        selected_topics, reassigned = human_checkpoint(non_personal, allow_reassign=True)

        # handle reassigned topics
        for target_project, reassigned_topics in reassigned.items():
            if target_project not in grouped:
                grouped[target_project] = []
            grouped[target_project].extend(reassigned_topics)
            processed_projects.add(target_project)
            print(f"  → Added {len(reassigned_topics)} topic(s) to {target_project}")

        if not selected_topics:
            print(f"  ↷ Skipping {project_name} — nothing selected")
            continue

        processed_projects.add(project_name)
        selected_per_project[project_name] = selected_topics

    # update grouped to reflect type corrections made in human_checkpoint
    for project_name, selected_topics in selected_per_project.items():
        grouped[project_name] = selected_topics

    # create folders now that all type overrides have been applied
    create_project_folders(grouped)

    # Pass 2: summarize each selected topic
    for project_name, selected_topics in selected_per_project.items():
        for topic in selected_topics:
            print(f"\n  Processing: [{topic['type']}] {topic['name']}")

            results = summarize_for_notes(new_chunk, topic, usage_rows)

            from config import ENABLE_ERRORS_CHECKPOINT
            if ENABLE_ERRORS_CHECKPOINT and "errors_fixes" in results:
                approved_errors = errors_checkpoint(results["errors_fixes"])
                results["approved_errors"] = approved_errors

            if project_name == "Personal":
                write_notes(results, "Personal")
            else:
                write_notes(results, project_name)
    
   # Step 7: one planning + learning note per project
    for project_name in grouped.keys():
        if project_name not in processed_projects:
            continue
        non_personal = [t for t in grouped[project_name]
                       if t["type"] != "PERSONAL"]
        if not non_personal:
            continue

        folder_name = project_name.replace(" ", "_")
        vault = Path(VAULT_PATH)

        # determine project type
        is_pure_exploration = all(t["type"] == "EXPLORATION" for t in non_personal)

        # snapshot before any writes
        if project_name != "Personal":
            print(f"\n  Snapshotting {project_name}...")
            snapshot_project(project_name)

        folder_name = project_name.replace(" ", "_")
        vault = Path(VAULT_PATH)

        # planning — skip for Personal and EXPLORATION
        if project_name != "Personal" and not is_pure_exploration:
            print(f"\n  Generating planning note for: {project_name}")
            planning_path = str(
                vault / folder_name / "Planning" / f"{folder_name}_Planning.md"
            )
            existing_planning = read_file(planning_path)
            planning_content = summarize_planning(new_chunk, project_name, existing_planning, usage_rows)
            write_file(planning_path, planning_content, mode="overwrite")
            print(f"  ✓ Planning note written for {project_name}")

        # exploration notes — run if any topic is EXPLORATION
        has_exploration = any(t["type"] == "EXPLORATION" for t in non_personal)
        if has_exploration and project_name != "Personal":
            print(f"\n  Generating exploration note for: {project_name}")
            exploration_path = str(
                vault / folder_name / f"{folder_name}_Exploration.md"
            )
            existing_exploration = read_file(exploration_path)
            exploration_content = summarize_exploration(new_chunk, project_name, existing_exploration, usage_rows)
            write_file(exploration_path, exploration_content, mode="overwrite")
            print(f"  ✓ Exploration note written for {project_name}")

        # learning — run for all projects including Personal
        has_learning = any(t["type"] == "LEARNING" for t in non_personal)
        if has_learning:
            print(f"\n  Generating learning note for: {project_name}")
            if project_name == "Personal":
                learning_path = str(vault / "Personal" / "Knowledge_Base.md")
            else:
                learning_path = str(
                    vault / folder_name / "Learning" / f"{folder_name}_Learning.md"
                )
            existing_learning = read_file(learning_path)
            learning_content = summarize_learning(new_chunk, project_name, existing_learning, usage_rows)
            write_file(learning_path, learning_content, mode="overwrite")
            print(f"  ✓ Learning note written for {project_name}")

        # current_state — skip for Personal and EXPLORATION
        if project_name != "Personal" and not is_pure_exploration:
            print(f"\n  Generating current_state for: {project_name}")
            state_path = str(vault / folder_name / "Code" / "current_state.md")
            existing_state = read_file(state_path)
            state_content = summarize_current_state(new_chunk, project_name, existing_state, usage_rows)
            write_file(state_path, state_content, mode="overwrite")
            print(f"  ✓ current_state.md written for {project_name}")

            # project_map — skip for Personal and EXPLORATION
            print(f"\n  Generating project_map for: {project_name}")
            map_path = str(vault / folder_name / "Code" / "project_map.md")
            existing_map = read_file(map_path)
            map_content = summarize_project_map(new_chunk, project_name, existing_map, usage_rows)
            write_file(map_path, map_content, mode="overwrite")
            print(f"  ✓ project_map.md written for {project_name}")

        # changes — skip for EXPLORATION
        if not is_pure_exploration:
            changes = extract_changes(new_chunk, project_name, usage_rows)
            if changes:
                folder_name_c = project_name.replace(" ", "_")
                project_path_c = Path(VAULT_PATH) / folder_name_c
                for change in changes:
                    change_name = change["name"].lower().replace(" ", "-")
                    change_path = project_path_c / "Code" / "changes" / change_name
                    if not change_path.exists():
                        change_path.mkdir(parents=True, exist_ok=True)
                        write_file(str(change_path / "proposal.md"), change["proposal"], mode="overwrite")
                        write_file(str(change_path / "design.md"), change["design"], mode="overwrite")
                        write_file(str(change_path / "tasks.md"), change["tasks"], mode="overwrite")
                        write_file(str(change_path / "spec_delta.md"), change["spec_delta"], mode="overwrite")
                        print(f"  ✓ changes/{change_name}/ written")
                    else:
                        print(f"  ↷ changes/{change_name}/ already exists — skipping")

    # Step 8: save anchor
    save_anchor(new_conversation, anchor_path)
    print(f"\n✓ Anchor updated")

    # Step 9: trim inbox file to anchor size
    anchor_content = read_file(anchor_path)
    write_file(inbox_file, anchor_content, mode="overwrite")
    print(f"✓ Inbox file trimmed to anchor size")

    print(f"\n{'='*60}")
    print("✓ DONE")
    print(f"{'='*60}")

    # Step 10: write usage log
    if usage_rows:
        append_usage_log(usage_rows)

def group_and_rename_projects(topics: list) -> dict:
    """
    Group detected topics by project name, show summary, and allow
    interactive renaming. Returns grouped dict with no folder creation.
    Called at Step 5 in process_conversation, before human_checkpoint.
    """
    from config import VAULT_PATH

    # Step 1: group topics by project
    grouped = {}
    for topic in topics:
        project_name = topic["project"]
        if project_name not in grouped:
            grouped[project_name] = []
        grouped[project_name].append(topic)

    # Step 2: show groups for human review
    print("\nDetected project groups:\n")
    project_names = list(grouped.keys())

    for i, project_name in enumerate(project_names, start=1):
        topics = grouped[project_name]
        print(f"  {i}. {project_name}")
        for topic in topics:
            print(f"       - [{topic['type']}] {topic['name']}")

    # Step 3: let human rename projects
    print("\nWould you like to rename any projects?")
    print("Enter number and new name (e.g. '2 LangGraph Agent') or 'done':")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() == "done":
            break
        try:
            parts = user_input.split(" ", 1)
            if len(parts) == 2:
                idx = int(parts[0]) - 1
                new_name = parts[1].strip()
                old_name = project_names[idx]
                project_names[idx] = new_name
                grouped[new_name] = grouped.pop(old_name)
                print(f"  ✓ Renamed '{old_name}' → '{new_name}'")
            else:
                print("  ⚠ Format: number + name, e.g. '2 LangGraph Agent'")
        except (ValueError, IndexError):
            print("  ⚠ Invalid input — try again or type 'done'")

    return grouped

def create_project_folders(grouped: dict) -> None:
    """
    Create vault folder structure for each project based on its topic types.
    EXPLORATION projects get inbox/ only.
    All other projects get full structure: inbox/, Code/changes/, Planning/, Learning/.
    Called after human_checkpoint so folder structure reflects corrected types.
    """
    from config import VAULT_PATH

    vault = Path(VAULT_PATH)

    print("\nSetting up project folders...")

    for project_name, topics in grouped.items():
        if project_name == "Personal":
            personal_path = vault / "Personal"
            if not personal_path.exists():
                personal_path.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created: Personal/")
            else:
                print(f"  ✓ Exists:  Personal/")
            continue

        folder_name = project_name.replace(" ", "_")
        project_path = vault / folder_name

        if project_path.exists():
            print(f"  ✓ Exists:  {folder_name}/")
            continue

        is_exploration = all(t["type"] == "EXPLORATION" for t in topics)

        if is_exploration:
            (project_path / "inbox").mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {folder_name}/ (exploration structure)")
        else:
            (project_path / "inbox").mkdir(parents=True, exist_ok=True)
            (project_path / "Code" / "changes").mkdir(parents=True, exist_ok=True)
            (project_path / "Planning").mkdir(parents=True, exist_ok=True)
            (project_path / "Learning").mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {folder_name}/")

    print("\n✓ All project folders ready")



if __name__ == "__main__":

    # -----------------------------------------------------------------------
    # Real run — comment out unit tests once verified
    # -----------------------------------------------------------------------

    # Run 1: LangGraph QC Agent
    # inbox_file = r"C:\Users\Yiyi.Luo\Notes\LangGraph_QC_Agent\inbox\2026-05-28.txt"
    # project_folder = r"C:\Users\Yiyi.Luo\Notes\LangGraph_QC_Agent"
    # process_conversation(inbox_file, project_folder)

    # Run 2: Personal Memory Agent  
    # inbox_file = r"C:\Users\Yiyi.Luo\Notes\Personal_Memory_Agent\inbox\2026-05-28.txt"
    # project_folder = r"C:\Users\Yiyi.Luo\Notes\Personal_Memory_Agent"
    # process_conversation(inbox_file, project_folder)


    # Run 3: Govwin API Exploration
    # inbox_file = r"C:\Users\Yiyi.Luo\Notes\GovWin_IQ_Pipeline\inbox\input.txt"
    # project_folder = r"C:\Users\Yiyi.Luo\Notes\GovWin_IQ_Pipeline"
    # process_conversation(inbox_file, project_folder)

    # Refresh code index (run when you want to update code_index.md)
    summarize_code_index("Personal Memory Agent")
    # summarize_code_index("LangGraph QC Agent")

    # Refresh file index (run when you want to update file_index.md for an exploration project)
    # summarize_file_index("GovWin IQ Pipeline", r"C:\Users\Yiyi.Luo\Notes\GovWin_IQ_Pipeline\Files")

    # Transition GovWin from exploration to technical
    # transition_to_technical("GovWin IQ Pipeline")
