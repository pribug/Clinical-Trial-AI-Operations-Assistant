# study_setup.py

"""
Clinical Trial AI Operations Assistant
--------------------------------------

Study Setup module containing:
1. Preliminary study specification generation
2. Deterministic protocol verification
3. Optional AI explanation of verification findings
4. Research-site feasibility scoring
5. Site feasibility email drafting

Design principle:
- Python performs deterministic validation and numerical calculations.
- Ollama is used for language generation/explanation only.
- Missing clinical information is never filled by Python.
- Human review is required before clinical, regulatory, or operational use.
"""

import json
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


# ============================================================
# MODEL
# ============================================================

OLLAMA_MODEL = "llama3.2:3b"


@st.cache_resource
def get_llm(model=OLLAMA_MODEL):
    return ChatOllama(
        model=model,
        temperature=0,
    )


# ============================================================
# CONSTANTS
# ============================================================

MISSING = "Not specified - requires human review."

DEMO_SITE_DATA = pd.DataFrame(
    {
        "Site_ID": ["S1", "S2", "S3", "S4"],
        "Country": ["India", "Singapore", "Malaysia", "Thailand"],
        "Monthly_Patients": [80, 40, 60, 35],
        "Active_Trials": [4, 2, 5, 1],
        "EDC_Experience": ["Yes", "Yes", "No", "Yes"],
        "Avg_Enrollment_Days": [45, 30, 55, 25],
    }
)

REQUIRED_SITE_COLUMNS = [
    "Site_ID",
    "Monthly_Patients",
    "Active_Trials",
    "EDC_Experience",
    "Avg_Enrollment_Days",
]

DURATION_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)?\s*"
    r"(?:day|days|week|weeks|month|months|year|years)\s*$",
    re.IGNORECASE,
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_study_state():
    defaults = {
        "study_idea": "",
        "study_specification": None,
        "protocol_verification": None,
        "protocol_explanation": None,
        "site_data": DEMO_SITE_DATA.copy(),
        "site_ranking": None,
        "site_explanation": None,
        "site_email": None,
        "study_audit": [],
        "target_patients": 60,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def is_missing_value(value):
    text = normalize_text(value)

    return (
        not text
        or "not specified" in text
        or "requires human review" in text
        or text in {"requires review", "not specified"}
    )


def safe_json_loads(text):
    """Parse normal JSON or JSON wrapped in markdown fences."""
    if not text:
        return None

    text = str(text).strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Conservative fallback: extract the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    return None


def add_study_audit(task, status, details=""):
    st.session_state["study_audit"].append(
        {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Task": task,
            "Status": status,
            "Details": details,
        }
    )


def contains_explicit_information(generated_value, study_idea):
    """
    Conservative check used for descriptive fields such as population.

    It is not a semantic validator. It only checks whether the generated
    content has meaningful overlap with the user's original text.
    """
    generated = normalize_text(generated_value)
    idea = normalize_text(study_idea)

    if not generated or not idea:
        return False

    if generated in idea:
        return True

    generated_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", generated))
    idea_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", idea))

    if not generated_words:
        return False

    overlap = generated_words & idea_words
    return len(overlap) / len(generated_words) >= 0.60


# ============================================================
# PROTOCOL GENERATION
# ============================================================

def generate_protocol(study_idea):
    """
    Generate a preliminary study specification.

    The LLM structures information supplied by the user.
    Python then sanitizes the result so unsupported clinical
    details are not presented as established facts.
    """

    llm = get_llm()

    system_prompt = """
You are a Clinical Research Planning Assistant.

Convert the user's study idea into a PRELIMINARY study specification.

This is NOT a clinical trial protocol.
This is NOT clinical, statistical, scientific, or regulatory validation.

CORE RULE:
Use ONLY information explicitly provided by the user.

If a field was not provided, return exactly:
"Not specified - requires human review."

DO NOT invent or infer:
- study design
- randomization
- blinding
- placebo
- comparator
- treatment arms
- treatment name
- dose
- treatment schedule
- eligibility criteria
- inclusion criteria
- exclusion criteria
- clinical endpoints
- validated measurement scales
- statistical analysis
- sample-size justification
- statistical power
- effect size
- variability
- dropout rate
- visit schedule
- follow-up schedule

FIELD RULES:

TITLE:
Create a concise title using only information present in the user's idea.

OBJECTIVE:
Summarize the purpose stated by the user.
Do not automatically add efficacy, safety, tolerability, effectiveness,
superiority, or non-inferiority unless the user stated that purpose.

STUDY DESIGN:
Only report a design explicitly stated by the user.

POPULATION:
Only report population characteristics explicitly stated by the user.
Do not add age limits, diagnostic confirmation, treatment history,
comorbidities, or eligibility criteria.

PRIMARY ENDPOINT:
Only report an endpoint explicitly stated by the user.
Preserve the user's wording.
Do not turn a general outcome into a named clinical scale.

SAMPLE SIZE:
If the user provides a participant number, record it as a TARGET SAMPLE SIZE.
Do not justify the number statistically.
If no participant number is provided, return the missing-information phrase.

STUDY DURATION:
Record a duration explicitly provided by the user.

KEY VISITS:
Only include visits explicitly stated by the user.
Do NOT create visits from the study duration.

ASSUMPTIONS:
Do NOT create assumptions merely because information is missing.
Return an empty list unless the user's wording itself requires a
specific interpretation.

Return ONLY valid JSON.
Do not use markdown.

Use exactly:
{
  "title": "",
  "objective": "",
  "study_design": "",
  "population": "",
  "primary_endpoint": "",
  "sample_size": "",
  "study_duration": "",
  "key_visits": [],
  "assumptions": []
}
"""

    human_prompt = f"""
User study idea:

{study_idea}

Create the preliminary study specification.
Use only information explicitly provided by the user.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        result = safe_json_loads(response.content)

        if result is None:
            return {
                "title": "Study specification requires review",
                "objective": MISSING,
                "study_design": MISSING,
                "population": MISSING,
                "primary_endpoint": MISSING,
                "sample_size": MISSING,
                "study_duration": MISSING,
                "key_visits": [],
                "assumptions": [
                    "The model did not return valid structured JSON."
                ],
            }

        defaults = {
            "title": MISSING,
            "objective": MISSING,
            "study_design": MISSING,
            "population": MISSING,
            "primary_endpoint": MISSING,
            "sample_size": MISSING,
            "study_duration": MISSING,
            "key_visits": [],
            "assumptions": [],
        }

        for field, default in defaults.items():
            if field not in result or result[field] is None:
                result[field] = default

        if not isinstance(result["key_visits"], list):
            result["key_visits"] = []

        if not isinstance(result["assumptions"], list):
            result["assumptions"] = [str(result["assumptions"])]

        return sanitize_protocol_result(result, study_idea)

    except Exception as exc:
        return {
            "title": "Protocol generation failed",
            "objective": MISSING,
            "study_design": MISSING,
            "population": MISSING,
            "primary_endpoint": MISSING,
            "sample_size": MISSING,
            "study_duration": MISSING,
            "key_visits": [],
            "assumptions": [f"LLM generation error: {exc}"],
        }


def sanitize_protocol_result(result, study_idea):
    """
    Deterministic safety layer after LLM generation.

    Important:
    This does not try to repair the LLM creatively.
    Unsupported information is removed/marked missing.
    """

    idea = normalize_text(study_idea)

    fields = [
        "title",
        "objective",
        "study_design",
        "population",
        "primary_endpoint",
        "sample_size",
        "study_duration",
    ]

    for field in fields:
        if field not in result or result[field] is None:
            result[field] = MISSING

    # --------------------------------------------------------
    # Sample size
    # --------------------------------------------------------

    sample_size = str(result["sample_size"]).strip()

    if not is_missing_value(sample_size):
        numbers = re.findall(r"\b\d[\d,]*\b", sample_size)

        if not numbers:
            result["sample_size"] = MISSING
        else:
            # Every numeric sample-size claim must be traceable to
            # the user's input.
            if not any(
                re.search(
                    rf"\b{re.escape(n.replace(',', ''))}\b",
                    idea,
                )
                for n in numbers
            ):
                result["sample_size"] = MISSING

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    duration = str(result["study_duration"]).strip()

    if not is_missing_value(duration):
        if not DURATION_PATTERN.fullmatch(duration):
            result["study_duration"] = MISSING
        elif duration.lower() not in idea:
            result["study_duration"] = MISSING

    # --------------------------------------------------------
    # Endpoint
    # --------------------------------------------------------

    endpoint = str(result["primary_endpoint"]).strip()

    if not is_missing_value(endpoint):
        if normalize_text(endpoint) not in idea:
            result["primary_endpoint"] = MISSING

    # --------------------------------------------------------
    # Study design
    # --------------------------------------------------------

    design = str(result["study_design"]).strip()

    if not is_missing_value(design):
        if normalize_text(design) not in idea:
            result["study_design"] = MISSING

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    population = str(result["population"]).strip()

    if not is_missing_value(population):
        if not contains_explicit_information(population, study_idea):
            result["population"] = MISSING

    # --------------------------------------------------------
    # Key visits
    # --------------------------------------------------------

    cleaned_visits = []

    for visit in result.get("key_visits", []):
        visit_text = str(visit).strip()

        if visit_text and normalize_text(visit_text) in idea:
            cleaned_visits.append(visit_text)

    result["key_visits"] = cleaned_visits

    # --------------------------------------------------------
    # Assumptions
    # --------------------------------------------------------

    # The safest behavior for this version:
    # assumptions are allowed only when the model explicitly returned them.
    # They are ALWAYS displayed as requiring human review.
    if not isinstance(result.get("assumptions"), list):
        result["assumptions"] = [str(result["assumptions"])]

    result["assumptions"] = [
        str(a).strip()
        for a in result["assumptions"]
        if str(a).strip()
    ]

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    objective = str(result["objective"]).strip()

    if not is_missing_value(objective):
        # Prevent unsupported efficacy/safety claims from being treated
        # as user intent. If the model adds them, fall back to the
        # user's original wording rather than inventing a correction.
        unsupported = [
            "efficacy",
            "safety",
            "tolerability",
            "effectiveness",
            "superiority",
            "non-inferiority",
        ]

        objective_lower = normalize_text(objective)

        added_unsupported = any(
            term in objective_lower and term not in idea
            for term in unsupported
        )

        if added_unsupported:
            result["objective"] = study_idea.strip()

    return result


# ============================================================
# PROTOCOL VERIFICATION
# ============================================================

def verify_protocol_python(protocol, study_idea):
    """Perform deterministic checks; no LLM is used here."""

    issues = []

    def add_issue(item, category, reason):
        issues.append(
            {
                "Item": item,
                "Category": category,
                "Review Required": "Yes",
                "Reason": reason,
            }
        )

    # --------------------------------------------------------
    # Missing information
    # --------------------------------------------------------

    required_fields = {
        "Study Design": "study_design",
        "Population": "population",
        "Primary Endpoint": "primary_endpoint",
        "Sample Size": "sample_size",
        "Study Duration": "study_duration",
    }

    for label, field in required_fields.items():
        if is_missing_value(protocol.get(field, "")):
            add_issue(
                label,
                "Missing Information",
                "This element was not sufficiently specified "
                "and requires human review.",
            )

    # --------------------------------------------------------
    # Unsupported design terminology
    # --------------------------------------------------------

    design = normalize_text(protocol.get("study_design", ""))
    idea = normalize_text(study_idea)

    design_terms = [
        "randomized",
        "randomised",
        "double blind",
        "double-blind",
        "single blind",
        "single-blind",
        "placebo",
        "controlled",
        "parallel group",
        "parallel-group",
        "crossover",
        "cross-over",
    ]

    unsupported_design = [
        term for term in design_terms
        if term in design and term not in idea
    ]

    if unsupported_design:
        add_issue(
            "Study Design",
            "AI-Inferred Information",
            "The generated study design contains unsupported "
            "design information: "
            + ", ".join(unsupported_design)
            + ".",
        )

    # --------------------------------------------------------
    # Sample size
    # --------------------------------------------------------

    sample_size = str(protocol.get("sample_size", "")).strip()

    if not is_missing_value(sample_size):
        numbers = re.findall(r"\b\d[\d,]*\b", sample_size)

        if numbers:
            user_numbers = {
                n.replace(",", "")
                for n in re.findall(r"\b\d[\d,]*\b", study_idea)
            }

            for number in numbers:
                if number.replace(",", "") not in user_numbers:
                    add_issue(
                        "Sample Size",
                        "AI-Inferred Information",
                        "The generated participant number was not "
                        "explicitly provided by the user.",
                    )
                    break

            add_issue(
                "Sample Size",
                "Statistical Review",
                "The participant number is treated only as a "
                "target enrollment. Statistical power, effect size, "
                "variability, significance level, and dropout "
                "assumptions have not been established.",
            )

    # --------------------------------------------------------
    # Endpoint
    # --------------------------------------------------------

    endpoint = str(
        protocol.get("primary_endpoint", "")
    ).strip()

    if not is_missing_value(endpoint):
        if normalize_text(endpoint) not in idea:
            add_issue(
                "Primary Endpoint",
                "AI-Inferred Information",
                "The generated endpoint was not explicitly "
                "provided in the original study idea and "
                "requires human review.",
            )

    # --------------------------------------------------------
    # Population
    # --------------------------------------------------------

    population = str(
        protocol.get("population", "")
    ).strip()

    if not is_missing_value(population):
        if not contains_explicit_information(population, study_idea):
            add_issue(
                "Population",
                "AI-Inferred Information",
                "The generated population description contains "
                "information that was not clearly specified by the user.",
            )

    # --------------------------------------------------------
    # Assumptions
    # --------------------------------------------------------

    assumptions = protocol.get("assumptions", [])

    if assumptions:
        add_issue(
            "Assumptions",
            "Human Review",
            f"{len(assumptions)} AI-generated assumption(s) "
            "require human review.",
        )

    # --------------------------------------------------------
    # Key visits
    # --------------------------------------------------------

    if not protocol.get("key_visits"):
        add_issue(
            "Key Visits",
            "Missing Information",
            "No visit schedule was explicitly provided. "
            "The study team must define the visit schedule.",
        )

    # --------------------------------------------------------
    # Overall disclaimer
    # --------------------------------------------------------

    add_issue(
        "Overall Study Specification",
        "Human Review",
        "This is an AI-assisted preliminary planning output. "
        "It has not been clinically, statistically, scientifically, "
        "or regulatorily validated.",
    )

    return pd.DataFrame(issues)


def explain_verification(verification_df):
    """
    Optional LLM summary of Python's findings.

    The LLM does not perform the checks.
    """

    if verification_df.empty:
        return (
            "No structured issues were detected by the automated checks. "
            "Human review is still required before clinical use."
        )

    llm = get_llm()

    review_text = verification_df[
        ["Item", "Category", "Reason"]
    ].to_string(index=False)

    system_prompt = """
You are a clinical research review assistant.

Summarize ONLY the automated findings supplied by Python.

Do not:
- add new clinical facts
- invent missing study information
- recommend a study design
- calculate sample size
- validate the study
- provide treatment recommendations

Return 3-5 concise bullet points focused only on what
the study team needs to review.
"""

    human_prompt = f"""
Automated verification findings:

{review_text}

Summarize the human-review points.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        return response.content.strip()
    except Exception as exc:
        return (
            "AI explanation unavailable. Review the structured "
            f"verification table above. Error: {exc}"
        )


# ============================================================
# SITE DATA VALIDATION
# ============================================================

def validate_site_data(df):
    if not isinstance(df, pd.DataFrame):
        return False, "The site dataset is not a valid table."

    missing = [
        column
        for column in REQUIRED_SITE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        return (
            False,
            "Missing required columns: " + ", ".join(missing),
        )

    if df.empty:
        return False, "The site dataset is empty."

    numeric_columns = [
        "Monthly_Patients",
        "Active_Trials",
        "Avg_Enrollment_Days",
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return False, f"'{column}' must contain numeric values."

        if df[column].isna().any():
            return False, f"'{column}' contains missing values."

        if (df[column] < 0).any():
            return False, f"'{column}' cannot contain negative values."

    if df["Monthly_Patients"].eq(0).all():
        return False, "'Monthly_Patients' cannot be zero for every site."

    return True, ""


# ============================================================
# SITE SCORING
# ============================================================

def normalize_positive(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(100.0, index=series.index)

    return ((series - minimum) / (maximum - minimum)) * 100


def normalize_negative(series):
    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(100.0, index=series.index)

    return ((maximum - series) / (maximum - minimum)) * 100


def calculate_site_scores(df, target_patients=None):
    """
    Calculate a transparent site feasibility score.

    Base score:
      Monthly Patients = 40%
      Enrollment Speed = 30%
      EDC Experience   = 20%
      Active Trials    = 10%

    If target_patients is supplied:
      Estimated Months Needed = target / monthly patients

    The target feasibility score contributes 20% of the
    study-specific Total Score, making the target actually
    affect the ranking.
    """

    valid, error = validate_site_data(df)

    if not valid:
        raise ValueError(error)

    scored = df.copy()

    scored["Patient_Score"] = normalize_positive(
        scored["Monthly_Patients"]
    )

    scored["Enrollment_Score"] = normalize_negative(
        scored["Avg_Enrollment_Days"]
    )

    scored["EDC_Score"] = (
        scored["EDC_Experience"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 100, "no": 0})
        .fillna(0)
    )

    scored["Trial_Load_Score"] = normalize_negative(
        scored["Active_Trials"]
    )

    scored["Base_Score"] = (
        scored["Patient_Score"] * 0.40
        + scored["Enrollment_Score"] * 0.30
        + scored["EDC_Score"] * 0.20
        + scored["Trial_Load_Score"] * 0.10
    )

    if target_patients is not None:
        target_patients = int(target_patients)

        if target_patients < 1:
            raise ValueError(
                "Target participant number must be at least 1."
            )

        scored["Estimated_Months_Needed"] = (
            target_patients
            / scored["Monthly_Patients"].replace(0, pd.NA)
        ).round(1)

        time_series = scored["Estimated_Months_Needed"]

        finite = time_series.dropna()

        if finite.empty:
            scored["Target_Feasibility_Score"] = 0.0
        else:
            best_time = finite.min()
            scored["Target_Feasibility_Score"] = (
                best_time / time_series
            ) * 100
            scored["Target_Feasibility_Score"] = (
                scored["Target_Feasibility_Score"]
                .fillna(0)
                .clip(0, 100)
            )

        scored["Total_Score"] = (
            scored["Base_Score"] * 0.80
            + scored["Target_Feasibility_Score"] * 0.20
        )
    else:
        scored["Total_Score"] = scored["Base_Score"]

    scored["Base_Score"] = scored["Base_Score"].round(1)
    scored["Total_Score"] = scored["Total_Score"].round(1)

    scored = (
        scored.sort_values(
            "Total_Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    scored["Rank"] = range(1, len(scored) + 1)

    return scored


def explain_site_ranking(scored_df, target_patients):
    if scored_df.empty:
        return "No site ranking is available for review."

    llm = get_llm()

    display_columns = [
        "Site_ID",
        "Monthly_Patients",
        "Active_Trials",
        "EDC_Experience",
        "Avg_Enrollment_Days",
        "Base_Score",
        "Total_Score",
        "Rank",
    ]

    if "Country" in scored_df.columns:
        display_columns.insert(1, "Country")

    if "Estimated_Months_Needed" in scored_df.columns:
        display_columns.append("Estimated_Months_Needed")

    ranking_data = scored_df[display_columns].to_string(index=False)

    system_prompt = """
You are a clinical trial site feasibility assistant.

Explain a ranking already calculated by Python.

Do not:
- recalculate scores
- change rankings
- invent site capabilities
- claim that a site is definitely suitable
- make regulatory claims

Explain strengths, trade-offs, the target-enrollment estimate,
and what should be confirmed by human feasibility review.

Estimated enrollment time is only a mathematical planning estimate.
Keep the explanation concise.
"""

    human_prompt = f"""
Target enrollment:
{target_patients} participants

Python-calculated ranking:

{ranking_data}

Explain:
1. Why the highest-ranked sites scored well.
2. Important trade-offs.
3. Why target enrollment affects the study-specific score.
4. What requires human feasibility confirmation.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        return response.content.strip()
    except Exception as exc:
        return f"AI ranking explanation unavailable: {exc}"


# ============================================================
# SITE EMAIL
# ============================================================

def generate_site_email(site_row, target_patients):
    llm = get_llm()

    site_id = site_row.get("Site_ID", "the site")
    country = site_row.get("Country", "")

    system_prompt = """
You are a clinical trial operations coordinator.

Draft a concise professional feasibility inquiry email.

Do not imply that the site has been selected.
Do not claim the site can meet enrollment.
Do not invent study details.

Ask the site to confirm:
- participant availability
- enrollment capacity
- expected enrollment timeline
- relevant operational feasibility information

Keep it professional and concise.
"""

    human_prompt = f"""
Site: {site_id}
Country: {country}
Target enrollment: {target_patients} participants

Draft the feasibility inquiry email.
"""

    try:
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )
        return response.content.strip()
    except Exception as exc:
        return f"Email generation unavailable: {exc}"


# ============================================================
# PROTOCOL UI
# ============================================================

def render_protocol_generator():
    st.header("Protocol Generator")

    st.info(
        "Generate a preliminary structured study specification. "
        "Human review is required before clinical use."
    )

    study_idea = st.text_area(
        "Enter your study idea",
        value=st.session_state.get("study_idea", ""),
        placeholder=(
            "Example: Evaluate a new topical treatment for "
            "psoriasis in 60 adults over 12 weeks."
        ),
        height=120,
        key="study_idea_input",
    )

    if st.button(
        "Generate Study Specification",
        key="generate_protocol",
    ):
        if not study_idea.strip():
            st.warning("Please enter a study idea first.")
            return

        with st.spinner("Generating preliminary study specification..."):
            protocol = generate_protocol(study_idea)

        st.session_state["study_idea"] = study_idea
        st.session_state["study_specification"] = protocol
        st.session_state["protocol_verification"] = None
        st.session_state["protocol_explanation"] = None

        add_study_audit(
            "Protocol Generation",
            "Generated - Pending Human Review",
            study_idea[:100],
        )

    protocol = st.session_state.get("study_specification")

    if not protocol:
        return

    st.subheader("Generated Study Specification")

    st.write(f"**Title:** {protocol.get('title', MISSING)}")
    st.write(f"**Objective:** {protocol.get('objective', MISSING)}")
    st.write(f"**Study Design:** {protocol.get('study_design', MISSING)}")
    st.write(f"**Population:** {protocol.get('population', MISSING)}")
    st.write(
        f"**Primary Endpoint:** "
        f"{protocol.get('primary_endpoint', MISSING)}"
    )
    st.write(
        f"**Target Sample Size:** "
        f"{protocol.get('sample_size', MISSING)}"
    )
    st.write(
        f"**Study Duration:** "
        f"{protocol.get('study_duration', MISSING)}"
    )

    if protocol.get("key_visits"):
        st.write("**Key Visits:**")
        for visit in protocol["key_visits"]:
            st.write(f"- {visit}")

    if protocol.get("assumptions"):
        st.warning(
            "AI-generated assumptions requiring human review"
        )
        for assumption in protocol["assumptions"]:
            st.write(f"- {assumption}")

    st.subheader("Automated Verification / Human Review")

    st.caption(
        "Python performs deterministic checks. AI is used only "
        "to summarize findings. This is not clinical or regulatory validation."
    )

    if st.button(
        "Run Verification Check",
        key="verify_protocol",
    ):
        with st.spinner("Running automated study checks..."):
            verification_df = verify_protocol_python(
                protocol,
                study_idea,
            )

        st.session_state["protocol_verification"] = verification_df

        if not verification_df.empty:
            with st.spinner("Generating concise review explanation..."):
                st.session_state["protocol_explanation"] = (
                    explain_verification(verification_df)
                )
        else:
            st.session_state["protocol_explanation"] = None

        add_study_audit(
            "Protocol Verification",
            "Automated Review - Human Review Required",
        )

    verification_df = st.session_state.get("protocol_verification")

    if verification_df is not None:
        if verification_df.empty:
            st.success(
                "No structured issues were detected. "
                "Human review is still required."
            )
        else:
            st.warning(
                f"{len(verification_df)} review item(s) identified."
            )
            st.dataframe(
                verification_df,
                use_container_width=True,
                hide_index=True,
            )

    explanation = st.session_state.get("protocol_explanation")

    if explanation:
        st.subheader("AI Review Explanation")
        st.info(explanation)


# ============================================================
# SITE FEASIBILITY UI
# ============================================================

def render_site_feasibility():
    st.header("Research Site Feasibility")

    st.info(
        "Rank sites using a transparent weighted scoring model. "
        "Python performs the calculations; AI explains the results."
    )

    st.subheader("Site Dataset")

    dataset_mode = st.radio(
        "Choose site data",
        [
            "Use Demo Dataset",
            "Upload Site Dataset",
        ],
        horizontal=True,
        key="site_dataset_mode",
    )

    if dataset_mode == "Use Demo Dataset":
        site_data = DEMO_SITE_DATA.copy()
        st.session_state["site_data"] = site_data

    else:
        uploaded_file = st.file_uploader(
            "Upload site feasibility Excel file",
            type=["xlsx", "xls"],
            key="site_dataset_upload",
        )

        if uploaded_file is None:
            st.caption(
                "Upload an Excel file containing site-feasibility information."
            )
            return

        try:
            site_data = pd.read_excel(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return

        valid, error = validate_site_data(site_data)

        if not valid:
            st.error(error)
            return

        st.session_state["site_data"] = site_data

    st.dataframe(
        site_data,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Study Requirements")

    # IMPORTANT:
    # Do not assign to st.session_state["target_patients"] here.
    # The widget owns that key after it is instantiated.
    target_patients = st.number_input(
        "Target number of participants",
        min_value=1,
        max_value=10000,
        value=st.session_state.get("target_patients", 60),
        step=1,
        key="target_patients",
    )

    st.caption(
        "The target affects estimated enrollment time and the "
        "study-specific site score."
    )

    protocol = st.session_state.get("study_specification")

    if protocol:
        col1, col2 = st.columns(2)

        with col1:
            st.write(
                "**Study:** "
                + str(protocol.get("title", MISSING))
            )

        with col2:
            st.write(
                "**Protocol target:** "
                + str(protocol.get("sample_size", MISSING))
            )

    if st.button(
        "Rank Sites",
        key="rank_sites",
    ):
        with st.spinner("Calculating site feasibility scores..."):
            try:
                scored = calculate_site_scores(
                    site_data,
                    target_patients=target_patients,
                )
            except Exception as exc:
                st.error(f"Could not calculate site scores: {exc}")
                return

        st.session_state["site_ranking"] = scored
        st.session_state["site_explanation"] = None
        st.session_state["site_email"] = None

        add_study_audit(
            "Site Feasibility",
            "Ranked - Pending Human Review",
            f"{target_patients} participants required",
        )

        with st.spinner("Generating AI explanation of site ranking..."):
            st.session_state["site_explanation"] = (
                explain_site_ranking(
                    scored,
                    target_patients,
                )
            )

    scored = st.session_state.get("site_ranking")

    if scored is None:
        return

    st.subheader("Site Ranking")

    result_columns = [
        "Rank",
        "Site_ID",
    ]

    if "Country" in scored.columns:
        result_columns.append("Country")

    result_columns += [
        "Monthly_Patients",
        "Active_Trials",
        "EDC_Experience",
        "Avg_Enrollment_Days",
        "Base_Score",
    ]

    if "Estimated_Months_Needed" in scored.columns:
        result_columns.append("Estimated_Months_Needed")

    if "Target_Feasibility_Score" in scored.columns:
        result_columns.append("Target_Feasibility_Score")

    result_columns.append("Total_Score")

    st.dataframe(
        scored[result_columns],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Base score: Monthly Patients 40% · Enrollment Speed 30% · "
        "EDC Experience 20% · Active Trials 10%. "
        "Target Feasibility contributes 20% to the study-specific score."
    )

    st.caption(
        "Estimated enrollment time is a mathematical planning estimate "
        "based only on the supplied monthly-patient value. It is not a guarantee."
    )

    explanation = st.session_state.get("site_explanation")

    if explanation:
        st.subheader("AI Feasibility Review")
        st.info(explanation)

    st.subheader("Sites for Further Feasibility Review")

    for _, row in scored.head(2).iterrows():
        country = (
            f" — {row['Country']}"
            if "Country" in row.index
            else ""
        )

        estimated = ""

        if "Estimated_Months_Needed" in row.index:
            estimated = (
                f"; estimated {row['Estimated_Months_Needed']} months "
                f"for target"
            )

        st.write(
            f"**#{int(row['Rank'])} {row['Site_ID']}{country}** — "
            f"Score: {row['Total_Score']}/100{estimated}"
        )

    st.subheader("Site Feasibility Email")

    if len(scored) > 0:
        site_options = scored["Site_ID"].tolist()

        selected_site = st.selectbox(
            "Select a site for the feasibility inquiry",
            site_options,
            key="selected_site_for_email",
        )

        selected_row = scored[
            scored["Site_ID"] == selected_site
        ].iloc[0]

        if st.button(
            "Generate Feasibility Email",
            key="generate_site_email",
        ):
            with st.spinner("Drafting site feasibility email..."):
                email = generate_site_email(
                    selected_row,
                    target_patients,
                )

            st.session_state["site_email"] = email

            add_study_audit(
                "Site Communication",
                "Draft Generated - Pending Human Review",
                f"Site: {selected_site}",
            )

        email = st.session_state.get("site_email")

        if email:
            st.text_area(
                "Email Draft",
                email,
                height=180,
            )

            st.caption(
                "Review and edit the draft before sending."
            )


# ============================================================
# MAIN RENDERER
# ============================================================

def render_study_setup():
    initialize_study_state()

    st.title("Clinical Trial AI Operations Assistant")

    st.write(
        "Use AI-assisted workflows for preliminary study planning "
        "and research-site feasibility assessment."
    )

    tab1, tab2 = st.tabs(
        [
            "Protocol Generator",
            "Site Feasibility",
        ]
    )

    with tab1:
        render_protocol_generator()

    with tab2:
        render_site_feasibility()

    st.divider()
    st.subheader("Study Setup Audit Log")

    audit = st.session_state.get("study_audit", [])

    if audit:
        audit_df = pd.DataFrame(audit)

        st.dataframe(
            audit_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Study Setup Audit Log",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name="study_setup_audit.csv",
            mime="text/csv",
        )
    else:
        st.caption("No Study Setup actions have been logged yet.")