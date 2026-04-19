from utils.open_questions import build_open_ended_question, classify_open_ended


def test_build_who_author() -> None:
    fact = {
        "subject": "Sonnet 109",
        "relation": "author",
        "subject_relation": "the author of Sonnet 109",
        "object_pri": "William Shakespeare",
        "object_ctx": "Upton Sinclair",
    }
    q = build_open_ended_question(fact)
    assert q == "Who is the author of Sonnet 109?"


def test_build_where_place_of_birth() -> None:
    fact = {
        "subject": "Florian Reichstädter",
        "relation": "place of birth",
        "subject_relation": "the place of birth of Florian Reichstädter",
        "object_pri": "Vienna",
        "object_ctx": "Philadelphia",
    }
    assert build_open_ended_question(fact) == "Where was Florian Reichstädter born?"


def test_build_what_capital() -> None:
    fact = {
        "subject": "Massachusetts",
        "relation": "capital",
        "subject_relation": "the capital of Massachusetts",
        "object_pri": "Boston",
        "object_ctx": "Rio de Janeiro",
    }
    assert build_open_ended_question(fact) == "What is the capital of Massachusetts?"


def test_build_what_occupation() -> None:
    fact = {
        "subject": "Wallace Dollase",
        "relation": "occupation",
        "subject_relation": "the occupation of Wallace Dollase",
        "object_pri": "horse trainer",
        "object_ctx": "lyricist",
    }
    assert build_open_ended_question(fact) == "What is the occupation of Wallace Dollase?"


def test_classify_memory_prior() -> None:
    fact = {
        "object_pri": "Boston",
        "object_ctx": "Rio de Janeiro",
        "possible_prior_answers": ["Boston, MA"],
        "object_ctx_aliases": [],
    }
    assert classify_open_ended("The capital is Boston.", "prior_yes", fact) == "memory"


def test_classify_context_ctx_row() -> None:
    fact = {
        "object_pri": "Boston",
        "object_ctx": "Rio de Janeiro",
        "possible_prior_answers": ["Boston"],
        "object_ctx_aliases": [],
    }
    assert classify_open_ended("Rio de Janeiro", "ctx_yes", fact) == "context"


def test_classify_short_color_word_boundary() -> None:
    fact = {
        "object_pri": "red",
        "object_ctx": "white",
        "possible_prior_answers": [],
        "object_ctx_aliases": [],
    }
    assert classify_open_ended("The color is red.", "prior_yes", fact) == "memory"
    assert classify_open_ended("redacted", "prior_yes", fact) == "other"
