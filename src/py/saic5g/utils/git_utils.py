import git
import configparser

def get_git_email():
    """Get user's git email

    This function is used to obtain the email associated with git for the
    current user. This is sometimes used for logging purposes.
    """
    r = git.Repo.init()
    reader = r.config_reader()
    try:
        email = reader.get_value("user", "email")
        return email
    except configparser.NoSectionError:
        raise ValueError(
            "Unable to obtain your user identifier automatically. \
                Please set an email in your git config. \
                git config --global user.email <your_email>"
        )
