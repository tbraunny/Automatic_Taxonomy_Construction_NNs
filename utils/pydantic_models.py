from typing import TypeVar, Generic, Literal, Optional, List
from pydantic import BaseModel, Field, field_validator

# Define a generic response model for structured LLM output.
T = TypeVar("T")


class LLMResponse(BaseModel, Generic[T]):
    answer: T

class TermDefinition(BaseModel):
    name: str = Field(..., description="The name of the function or component.")
    definition: Optional[str] = Field(
        None, description="A brief, succinct definition of this term."
    )

# Define a model for TrainingSingle details.
class TrainingSingleDetails(BaseModel):
    batch_size: Optional[int] = Field(
        None, description="The batch size used in the training step."
    )
    learning_rate_decay: Optional[float] = Field(
        None, description="The learning rate decay applied during training."
    )
    learning_rate_decay_epochs: Optional[int] = Field(
        None,
        description="The number of epochs after which learning rate decay is applied (optional).",
    )
    number_of_epochs: Optional[int] = Field(
        None, description="The total number of epochs used in training."
    )


class TrainingSingleResponse(LLMResponse[TrainingSingleDetails]):
    pass


# Define a model for the optimizer details.
class TrainingOptimizerDetails(BaseModel):
    # Data properties
    learning_rate: float = Field(
        ..., description="The learning rate used by the optimizer."
    )
    momentum: float = Field(
        ..., description="The momentum value used by the optimizer."
    )
    # Subclass
    subclass: Literal[
        "AdaDelta",
        "AdaGrad",
        "AdaGradDA",
        "Adam",
        "Ftrl",
        "GradientDescent",
        "Momentum",
        "ProximalAdaGrad",
        "ProximalGradientDescent",
        "RMSProp",
        "UNKNOWN",
    ] = Field(..., description="Name of the training optimizer.")


# Now define a specialized response type using the generic interface.
class TrainingOptimizerResponse(LLMResponse[TrainingOptimizerDetails]):
    pass


"""Define Dataset Models"""
# Define a model to represent the dataset details.

# class DataTypeDetails(BaseModel):
#     # Datatype Subclass
#     subclass: Literal[
#         "Image",
#         "MultiDimensionalCube",
#         "Text",
#         "Video"] = Field(..., description="The type of data present in the dataset.")

# class DatasetDetails(BaseModel):
#     # Data Properties
#     data_description: str = Field(..., description="A brief description of the dataset.")
#     data_doi: Optional[str] = Field(None, description="The DOI of the dataset, if available.")
#     data_location: Optional[str] = Field(None, description="The physical or digital location of the dataset.")
#     data_sample_dimensionality: Optional[str] = Field(None, description="The dimensionality (or shape) of a single data sample.")
#     data_sample_features: Optional[str] = Field(None, description="The features or attributes present in each data sample.")
#     data_samples: Optional[int] = Field(None, description="The total number of data samples in the dataset.")
#     is_transient_dataset: Optional[bool] = Field(None, description="Whether the dataset is transient (temporary) or persistent.")
#     # Connected classes
#     dataType: DataTypeDetails

# # Create a specialized response model for dataset processing.
# class DatasetResponse(LLMResponse[DatasetDetails]):
#     pass


# class DataTypeDetails(BaseModel):
#     subclass: str = Field(
#         ...,
#         description="The type of data present in the dataset. Suggested values: 'Image', 'MultiDimensionalCube', 'Text', 'Video'.",
#     )

class DatasetDetails(BaseModel):
    data_description: str = Field(
        ..., description="A brief description of the dataset."
    )
    dataset_name: Optional[str] = Field(None, description="The name of the dataset.")
    data_doi: Optional[str] = Field(
        None, description="The DOI of the dataset, if available."
    )
    # data_location: Optional[str] = Field(None, description="The physical or digital location of the dataset.")
    data_sample_dimensionality: Optional[str] = Field(
        None, description="The dimensionality (or shape) of a single data sample."
    )
    # data_sample_features: Optional[List[str]] = Field(None, description="The features or attributes present in each data sample.")
    data_samples: Optional[int] = Field(
        None, description="The total number of data samples in the dataset."
    )
    # is_transient_dataset: Optional[bool] = Field(None, description="Whether the dataset is transient (temporary) or persistent.")
    number_of_classes: Optional[int] = Field(
        None, description="The number of classes in the dataset."
    )
    data_type: str = Field(
        ...,
        description="The type of data present in the dataset. Suggested types: 'Image', 'MultiDimensionalCube', 'Text', 'Video'.",
    )

# Allows multiple datasets in the response
class MultiDatasetResponse(BaseModel):
    answer: List[DatasetDetails]


"""Define Process Objective Function"""


# # Example
# ACCEPTABLE_LOSS_FUNCTIONS = ["CrossEntropy", "MSE"]
# ACCEPTABLE_REGULARIZERS = ["L1Regularization", "L2Regularization", "L1L2Regularization"]

# class LossFunctionDetails(BaseModel):
#     type: str = Field(..., description="Name of the loss function associated with the given network.")

#     @field_validator("type")
#     def validate_loss_function(cls, value):
#         if value not in ACCEPTABLE_LOSS_FUNCTIONS:
#             print(f"Warning: '{value}' is not in the predefined list but will be allowed.")
#         return value

# class RegularFunctionDetails(BaseModel):
#     type: Optional[str] = Field(None, description="The regularizer function name associated with the given objective function.")

#     @field_validator("type")
#     def validate_regular_function(cls, value):
#         if value not in ACCEPTABLE_REGULARIZERS:
#             print(f"Warning: '{value}' is not in the predefined list but will be allowed.")
#         return value

# class CostFunctionDetails(BaseModel):
#     lossFunction: LossFunctionDetails
#     regularFunction: RegularFunctionDetails

# class ObjectiveFunctionDetails(BaseModel):
#     cost_function: CostFunctionDetails
#     objectiveFunction: Literal[
#         "minimize",
#         "maximize"
#     ] = Field(..., description="The objective type ('minimize' or 'maximize')")


# class ObjectiveFunctionResponse(LLMResponse[ObjectiveFunctionDetails]):
#     pass


# includes defintion of loss and regularizer
class ObjectiveFunctionDetails(BaseModel):
    loss: TermDefinition = Field(
        ..., description="Details about the loss function used."
    )
    regularizer: TermDefinition = Field(
        ..., description="Details about the regularizer used; if none put 'None'."
    )
    objective: Literal["minimize", "maximize"] = Field(
        ..., description="The optimization direction ('minimize' or 'maximize')."
    )


# class ObjectiveFunctionDetails(BaseModel):
#     loss: str = Field(..., description="Name of the loss function associated with the given network.")
#     regularizer: Optional[str] = Field( default=None, description="The regularizer function name associated with the loss function you provide, if any.")
#     objective: Literal["minimize", "maximize"] = Field(..., description="The objective type of the loss function ('minimize' or 'maximize')")


class ObjectiveFunctionResponse(ObjectiveFunctionDetails):
    pass

"""Define Process Task Characterization"""

class TaskCharacterizationDetails(BaseModel):
    task_type: Literal["Adversarial",
                "Self-Supervised Classification",
                "Semi-Supervised Classification",
                "Supervised Classification",
                "Unsupervised Classification",
                "Discrimination",
                "Generation",
                "Clustering",
                "Regression"] = Field(
        ..., description="Details about the training task type used."
    )

class TaskCharacterizationResponse(LLMResponse[TaskCharacterizationDetails]):
    pass

class Subnetwork(BaseModel):
    name: str = Field(..., description="Name or functional label of the subnetwork.")
    is_independent: bool = Field(
        ...,
        description="Whether the subnetwork has its own loss function or is trained independently."
    )

class Architecture(BaseModel):
    architecture_name: str = Field(..., description="Name of the neural network architecture.")
    subnetworks: List[Subnetwork] = Field(..., description="List of subnetworks within the architecture.")

# class Architecture(BaseModel):
#     architecture_name: str = Field(..., description="Explicit name of the neural network architecture.")
#     subnetworks: List[Subnetwork] = Field(
#         ..., description="List of subnetworks or functional components in this architecture."
#     )


class NetworkDetails(BaseModel):
    architectures: List[Architecture] = Field(
        ..., description="List of neural network architectures and their corresponding subnetworks."
    )

class NetworkResponse(LLMResponse[NetworkDetails]):
    pass