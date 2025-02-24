import pandas as pd


def test_exact_match(recommender):
    # Test exact match
    result = recommender.standardize_title("Movie 14", year=2010)
    assert isinstance(result, pd.Index)
    assert len(result) == 1


def test_case_insensitive(recommender):
    # Test case insensitivity
    result = recommender.standardize_title("movie 14", year=2010)
    assert isinstance(result, pd.Index)
    assert len(result) == 1


def test_misspelled_title(recommender):
    # Test with misspelled title
    result = recommender.standardize_title("mobie 1", year=2010)
    assert isinstance(result, dict)
    assert result["error"] == "Ambiguous or no match found"
    assert "movie 1" in [match[0] for match in result["possible_matches"]]


def test_multiple_versions_without_year(recommender):
    # Test movie with multiple versions without specifying year
    result = recommender.standardize_title("Movie 10")
    assert isinstance(result, dict)
    assert result["error"] == "Ambiguous or no match found"
    assert len(result["possible_matches"]) > 1


def test_multiple_versions_with_year(recommender):
    # Test movie with multiple versions with specific year
    result = recommender.standardize_title("Movie 10", year=2015)
    assert isinstance(result, pd.Index)
    assert len(result) == 1


def test_severely_misspelled(recommender):
    # Test with severely misspelled title
    result = recommender.standardize_title("msiovie 2")
    assert isinstance(result, dict)
    assert result["error"] == "Ambiguous or no match found"


def test_nonexistent_movie(recommender):
    # Test with completely nonexistent movie
    result = recommender.standardize_title("ThisMovieDoesNotExist123")
    assert isinstance(result, dict)
    assert result["error"] == "Ambiguous or no match found"
    assert len(result["possible_matches"]) == 0


def test_empty_title(recommender):
    # Test with empty string
    result = recommender.standardize_title("")
    assert isinstance(result, dict)
    assert result["error"] == "No movie title provided"


def test_invalid_year(recommender):
    # Test with valid title but non-existent year
    result = recommender.standardize_title("Movie 37", year=1900)
    assert isinstance(result, dict)
    assert result["error"] == "Ambiguous or no match found"
    assert result["message"] == "No exact match found for that title and year. Check the year and try again."


def test_valid_title_no_year(recommender):
    # Test with valid title but without year for a discoverable movie
    result = recommender.standardize_title("Movie 37")
    assert isinstance(result, pd.Index)
    assert len(result) == 1
