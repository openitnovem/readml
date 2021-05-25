import os


def get_interp_env():
    return os.getenv("INTERP_ENV", "local")


env = get_interp_env()
