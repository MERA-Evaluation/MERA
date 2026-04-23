class OpenJudgeError(Exception):
    pass


class JudgeConfigError(OpenJudgeError):
    pass


class JudgeBackendError(OpenJudgeError):
    pass


class JudgeParseError(OpenJudgeError):
    def __init__(
        self,
        message,
        doc_index=None,
        criterion_key=None,
        response_text=None,
    ):
        super().__init__(message)
        self.doc_index = doc_index
        self.criterion_key = criterion_key
        self.response_text = response_text
