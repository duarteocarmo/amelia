import pytest

from amelia_evals.runner import extract_answer


@pytest.mark.parametrize(
    argnames=("response", "letters", "expected"),
    argvalues=(
        (r"A resposta é \boxed{B}.", "ABCD", "B"),
        (r"A resposta é \boxed{b}.", "ABCD", "B"),
        (r"Primeiro \boxed{A}, finalmente \boxed{C}.", "ABCD", "C"),
        ("The options are A, B, C, or D.", "ABCD", None),
        ("A resposta correta é B.", "ABCD", None),
        ("B", "ABCD", None),
        (r"\boxed{D}", "ABC", None),
    ),
)
def test_extract_answer(response: str, letters: str, expected: str | None) -> None:
    assert extract_answer(response=response, letters=letters) == expected
