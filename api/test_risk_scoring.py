from main import RiskProfile, compute_clinical_risk


def test_no_risk_factors_returns_low():
    profile = RiskProfile()
    level, threshold = compute_clinical_risk(profile)
    assert level == "low"
    assert threshold == 0.50


def test_family_history_skin_cancer_alone_is_medium():
    profile = RiskProfile(family_history_skin_cancer=True)
    level, threshold = compute_clinical_risk(profile)
    assert level == "medium"
    assert threshold == 0.42


def test_family_history_and_atypical_moles_is_high():
    profile = RiskProfile(family_history_skin_cancer=True, atypical_moles=True)
    level, threshold = compute_clinical_risk(profile)
    assert level == "high"
    assert threshold == 0.35


def test_phototype_I_adds_score():
    profile = RiskProfile(skin_phototype="I")
    level, _ = compute_clinical_risk(profile)
    assert level == "medium"


def test_phototype_III_no_extra_score():
    profile = RiskProfile(skin_phototype="III")
    level, threshold = compute_clinical_risk(profile)
    assert level == "low"
    assert threshold == 0.50


def test_age_over_50_adds_score():
    profile = RiskProfile(age=55, family_history_other_cancer=True)
    level, _ = compute_clinical_risk(profile)
    assert level == "medium"


def test_all_factors_high():
    profile = RiskProfile(
        family_history_skin_cancer=True,
        atypical_moles=True,
        many_moles=True,
        skin_phototype="I",
        age=60,
    )
    level, threshold = compute_clinical_risk(profile)
    assert level == "high"
    assert threshold == 0.35
