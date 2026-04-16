class RecallError(Exception):
    """Base exception for recall."""


class DimensionMismatchError(RecallError):
    """Raised when the active embedder dimension does not match DB metadata."""

    def __init__(
        self,
        db_dimension: int,
        requested_dimension: int,
        db_provider: str,
        requested_provider: str,
        db_model: str,
        requested_model: str,
    ) -> None:
        self.db_dimension = db_dimension
        self.requested_dimension = requested_dimension
        self.db_provider = db_provider
        self.requested_provider = requested_provider
        self.db_model = db_model
        self.requested_model = requested_model
        super().__init__(
            "Embedding dimension mismatch: "
            f"DB is {db_dimension} ({db_provider}:{db_model}), "
            f"requested {requested_dimension} ({requested_provider}:{requested_model}). "
            "Run `recall rebuild-index --path <db-path> --namespace <namespace> ...` "
            "to migrate embeddings."
        )
