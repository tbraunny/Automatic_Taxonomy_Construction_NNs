import pytest
from pydantic import ValidationError
from utils.pydantic_models import DatasetDetails, DataTypeDetails


def test_dataset_valid():
    """Test valid dataset details initialization."""
    dataset = DatasetDetails(
        data_description="A dataset containing blah.",
        data_doi="10.1234/example-dataset",
        data_location="https://example.com/dataset",
        data_sample_dimensionality="512x512",
        data_sample_features="Word embeddings and metadata",
        data_samples=100000,
        is_transient_dataset=False,
        dataType=DataTypeDetails(subclass="Text"),
    )

    assert dataset.data_description == "A dataset containing blah."
    assert dataset.data_doi == "10.1234/example-dataset"
    assert dataset.data_location == "https://example.com/dataset"
    assert dataset.data_sample_dimensionality == "512x512"
    assert dataset.data_sample_features == "Word embeddings and metadata"
    assert dataset.data_samples == 100000
    assert dataset.is_transient_dataset is False
    assert isinstance(dataset.dataType, DataTypeDetails)
    assert dataset.dataType.subclass == "Text"


def test_dataset_optional_fields():
    """Test dataset initialization with optional fields missing."""
    dataset = DatasetDetails(
        data_description="A dataset with missing optional fields.",
        data_samples=5000,
        dataType=DataTypeDetails(subclass="Image"),
    )

    assert dataset.data_description == "A dataset with missing optional fields."
    assert dataset.data_samples == 5000
    assert dataset.data_doi is None
    assert dataset.data_location is None
    assert dataset.data_sample_dimensionality is None
    assert dataset.data_sample_features is None
    assert dataset.is_transient_dataset is None
    assert isinstance(dataset.dataType, DataTypeDetails)
    assert dataset.dataType.subclass == "Image"


def test_dataset_non_boolean_transient():
    """Test dataset with a non-boolean is_transient_dataset value."""
    with pytest.raises(ValidationError, match="value could not be parsed to a boolean"):
        DatasetDetails(
            data_description="Incorrect transient dataset type.",
            data_samples=100,
            is_transient_dataset="yes",
            dataType=DataTypeDetails(subclass="Image"),
        )


if __name__ == "__main__":
    pytest.main()
