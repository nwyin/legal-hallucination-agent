import datetime


def get_timestamp() -> str:
    return datetime.datetime.now().isoformat()
