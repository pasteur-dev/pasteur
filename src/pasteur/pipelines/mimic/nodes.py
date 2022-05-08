"""
This is a boilerplate pipeline 'mimic'
generated using Kedro 0.18.0
"""


def select_id(data):
    return data["subject_id"].to_frame()
