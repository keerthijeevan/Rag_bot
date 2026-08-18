"""
Payload index creation.

Payload indexes speed up Qdrant's server-side filtering on specific
metadata fields. This is distinct from query-time filter construction,
which lives in rag/retrieval/filters.py.

This module ensures that the payload indexes required by retrieval
filters exist with the correct schema.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Payload fields used for retrieval filtering
# ------------------------------------------------------------------

INDEXED_KEYWORD_FIELDS = [
    "source_type",   # pdf | docx | txt | markdown | csv | web
    "document_id",
    "category",
    "content_type",
    "url",
]

INDEXED_INTEGER_FIELDS = [
    "page",
]


# ------------------------------------------------------------------
# Payload index management
# ------------------------------------------------------------------

def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
) -> None:
    """
    Ensure all required payload indexes exist with the correct schema.

    Missing indexes are created.

    Existing indexes with the correct type are left unchanged.

    Existing indexes with the wrong type raise an error instead of
    silently hiding the configuration problem.
    """

    collection_info = client.get_collection(collection_name)
    existing_indexes = collection_info.payload_schema

    expected_indexes = {
        **{
            field: qmodels.PayloadSchemaType.KEYWORD
            for field in INDEXED_KEYWORD_FIELDS
        },
        **{
            field: qmodels.PayloadSchemaType.INTEGER
            for field in INDEXED_INTEGER_FIELDS
        },
    }

    for field_name, expected_type in expected_indexes.items():

        existing = existing_indexes.get(field_name)

        # ----------------------------------------------------------
        # Index already exists
        # ----------------------------------------------------------
        if existing is not None:
            actual_type = existing.data_type

            if actual_type != expected_type:
                raise RuntimeError(
                    f"Qdrant payload index '{field_name}' has type "
                    f"{actual_type}, but the application expects "
                    f"{expected_type}. Fix the collection index before "
                    f"continuing."
                )

            logger.debug(
                "Payload index '%s' already exists with correct type %s",
                field_name,
                expected_type,
            )

            continue

        # ----------------------------------------------------------
        # Index does not exist → create it
        # ----------------------------------------------------------
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=expected_type,
        )

        logger.info(
            "Created payload index '%s' with type %s",
            field_name,
            expected_type,
        )

    logger.info(
        "Verified payload indexes on collection '%s'",
        collection_name,
    )
