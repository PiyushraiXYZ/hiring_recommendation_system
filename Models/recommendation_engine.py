def generate_final_score(
    stability,
    long_term,
    monetary,
    culture
):

    final_score = (
        stability * 0.30 +
        long_term * 0.30 +
        monetary * 0.20 +
        culture * 0.20
    )

    return round(final_score, 2)
