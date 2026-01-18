from ..utils.download import DS

physio = "requires credentials and license from https://physionet.org"
rfel = "See https://relational.fel.cvut.cz/about for citation and license info."

datasets = {
    # Physionet
    "mimic_iv_1_0": DS(
        "https://physionet.org/files/mimiciv/1.0/", "mimiciv_1_0", True, physio
    ),
    "mimic_iv_2_0": DS(
        "https://physionet.org/files/mimiciv/2.0/", "mimiciv_2_0", True, physio
    ),
    "mimic_iv": DS(
        "https://physionet.org/files/mimiciv/2.2/", "mimiciv_2_2", True, physio
    ),
    "eicu": DS("https://physionet.org/files/eicu-crd/2.0/", "eicu_2_0", True, physio),
    "mimic_iv_ed": DS(
        "https://physionet.org/files/mimic-iv-ed/2.0/",
        "mimiciv_ed_2_0",
        True,
        physio,
    ),
    # https://relational.fel.cvut.cz/
    "rfel.ConsumerExpenditures": DS("relational.fel:ConsumerExpenditures", "rfel.consumerexp", False, rfel),
    "rfel.StudentLoan": DS("relational.fel:Student_loan", "rfel.studentloan", False, rfel),
    # SDGym
    "sdgym": DS(
        "s3:sdv-datasets",
        desc="license MIT (not clear if that applies to data), requires boto3 package",
    ),
}