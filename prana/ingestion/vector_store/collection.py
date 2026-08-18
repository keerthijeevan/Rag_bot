"""
Qdrant collection setup.

Owns the decision of what the collection's vector configuration looks
like (named dense vector + named sparse vector) so it stays in exactly
one place and cannot drift from what the embedders actually produce.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Named vector names
# ------------------------------------------------------------------

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ------------------------------------------------------------------
# Collection creation
# ------------------------------------------------------------------

def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    """
    Ensure the Qdrant collection exists with the required dense and
    sparse vector configuration.

    If the collection does not exist, it is created.

    If it already exists, its configuration is verified instead of
    blindly assuming that it is correct.
    """

    existing = {
        collection.name
        for collection in client.get_collections().collections
    }

    # --------------------------------------------------------------
    # Collection already exists
    # --------------------------------------------------------------

    if collection_name in existing:
        logger.info(
            "Qdrant collection '%s' already exists; verifying configuration",
            collection_name,
        )

        if not verify_collection_config(
            client=client,
            collection_name=collection_name,
            expected_dense_dim=dense_dim,
            expected_distance=distance,
        ):
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' does not match "
                "the application's expected vector configuration."
            )

        return

    # --------------------------------------------------------------
    # Collection does not exist → create it
    # --------------------------------------------------------------

    logger.info(
        "Creating Qdrant collection '%s' (dense_dim=%d)",
        collection_name,
        dense_dim,
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: qmodels.VectorParams(
                size=dense_dim,
                distance=distance,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(
                    on_disk=False,
                ),
            ),
        },
    )

    logger.info(
        "Qdrant collection '%s' created successfully",
        collection_name,
    )


# ------------------------------------------------------------------
# Collection configuration verification
# ------------------------------------------------------------------

def verify_collection_config(
    client: QdrantClient,
    collection_name: str,
    expected_dense_dim: int,
    expected_distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> bool:
    """
    Verify that an existing Qdrant collection matches the vector
    configuration required by this application.

    Checks:

    1. Named dense vector exists.
    2. Dense vector has the expected dimension.
    3. Dense vector uses the expected distance metric.
    4. Named sparse vector exists.

    Returns True when everything matches.
    Returns False when the collection is incompatible.
    """

    info = client.get_collection(collection_name)
    params = info.config.params

    # --------------------------------------------------------------
    # Dense vector configuration
    # --------------------------------------------------------------

    vectors_config = params.vectors

    if not isinstance(vectors_config, dict):
        logger.error(
            "Collection '%s' does not use the expected named-vector "
            "configuration.",
            collection_name,
        )
        return False

    dense_config = vectors_config.get(DENSE_VECTOR_NAME)

    if dense_config is None:
        logger.error(
            "Collection '%s' is missing the expected '%s' vector.",
            collection_name,
            DENSE_VECTOR_NAME,
        )
        return False

    # Check dense dimension.
    if dense_config.size != expected_dense_dim:
        logger.error(
            "Collection '%s' dense dimension mismatch: "
            "expected %d, found %d.",
            collection_name,
            expected_dense_dim,
            dense_config.size,
        )
        return False

    # Check dense distance metric.
    if dense_config.distance != expected_distance:
        logger.error(
            "Collection '%s' dense distance mismatch: "
            "expected %s, found %s.",
            collection_name,
            expected_distance,
            dense_config.distance,
        )
        return False

    # --------------------------------------------------------------
    # Sparse vector configuration
    # --------------------------------------------------------------

    sparse_config = getattr(params, "sparse_vectors", None)

    if not isinstance(sparse_config, dict):
        logger.error(
            "Collection '%s' does not contain the expected named "
            "sparse-vector configuration.",
            collection_name,
        )
        return False

    if SPARSE_VECTOR_NAME not in sparse_config:
        logger.error(
            "Collection '%s' is missing the expected '%s' sparse vector.",
            collection_name,
            SPARSE_VECTOR_NAME,
        )
        return False

    # --------------------------------------------------------------
    # Everything matches
    # --------------------------------------------------------------

    logger.info(
        "Qdrant collection '%s' configuration verified successfully "
        "(dense=%d, sparse=%s).",
        collection_name,
        expected_dense_dim,
        SPARSE_VECTOR_NAME,
    )

    return True
