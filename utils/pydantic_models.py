from typing import TypeVar, Generic, Literal, Optional
from pydantic import BaseModel, Field, field_validator

# Define a generic response model for structured LLM output.
T = TypeVar("T")
class LLMResponse(BaseModel, Generic[T]):
    answer: T

# Define a model for TrainingSingle details.
class TrainingSingleDetails(BaseModel):
    batch_size: int = Field(..., description="The batch size used in the training step.")
    learning_rate_decay: float = Field(..., description="The learning rate decay applied during training.")
    learning_rate_decay_epochs: Optional[int] = Field(
        None, description="The number of epochs after which learning rate decay is applied (optional)."
    )
    number_of_epochs: int = Field(..., description="The total number of epochs used in training.")

# Now define a specialized response type using the generic interface.
class TrainingSingleResponse(LLMResponse[TrainingSingleDetails]):
    pass

# Define a model for the optimizer details.
class TrainingOptimizerDetails(BaseModel):
    # Data properties
    learning_rate: float = Field(..., description="The learning rate used by the optimizer.")
    momentum: float = Field(..., description="The momentum value used by the optimizer.")
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

class DataTypeDetails(BaseModel):
    # Datatype Subclass
    subclass: Literal[
        "Image",
        "MultiDimensionalCube",
        "Text",
        "Video"] = Field(..., description="The type of data present in the dataset.")
    
class DatasetDetails(BaseModel):
    # Data Properties
    data_description: str = Field(..., description="A brief description of the dataset.")
    data_doi: Optional[str] = Field(None, description="The DOI of the dataset, if available.")
    data_location: Optional[str] = Field(None, description="The physical or digital location of the dataset.")
    data_sample_dimensionality: Optional[str] = Field(None, description="The dimensionality (or shape) of a single data sample.")
    data_sample_features: Optional[str] = Field(None, description="The features or attributes present in each data sample.")
    data_samples: Optional[int] = Field(None, description="The total number of data samples in the dataset.")
    is_transient_dataset: Optional[bool] = Field(None, description="Whether the dataset is transient (temporary) or persistent.")
    # Connected classes
    dataType: DataTypeDetails

# Create a specialized response model for dataset processing.
class DatasetResponse(LLMResponse[DatasetDetails]):
    pass

"""Define Dataset Models"""
# Define a model to represent the dataset details.

class DataTypeDetails(BaseModel):
    # Datatype Subclass
    subclass: Literal[
        "Image",
        "MultiDimensionalCube",
        "Text",
        "Video"] = Field(..., description="The type of data present in the dataset.")
    
class DatasetDetails(BaseModel):
    # Data Properties
    data_description: str = Field(..., description="A brief description of the dataset.")
    data_doi: Optional[str] = Field(None, description="The DOI of the dataset, if available.")
    data_location: Optional[str] = Field(None, description="The physical or digital location of the dataset.")
    data_sample_dimensionality: Optional[str] = Field(None, description="The dimensionality (or shape) of a single data sample.")
    data_sample_features: Optional[str] = Field(None, description="The features or attributes present in each data sample.")
    data_samples: Optional[int] = Field(None, description="The total number of data samples in the dataset.")
    is_transient_dataset: Optional[bool] = Field(None, description="Whether the dataset is transient (temporary) or persistent.")
    # Connected classes
    dataType: DataTypeDetails

# Create a specialized response model for dataset processing.
class DatasetResponse(LLMResponse[DatasetDetails]):
    pass

"""Define Process Objective Function"""

# Example
ACCEPTABLE_LOSS_FUNCTIONS = ["CrossEntropy", "MSE"]
ACCEPTABLE_REGULARIZERS = ["L1Regularization", "L2Regularization", "L1L2Regularization"]

class LossFunctionDetails(BaseModel):
    type: str = Field(..., description="Name of the loss function used in training.")

    @field_validator("type")
    def validate_loss_function(cls, value):
        if value not in ACCEPTABLE_LOSS_FUNCTIONS:
            print(f"Warning: '{value}' is not in the predefined list but will be allowed.")  
        return value

class RegularFunctionDetails(BaseModel):
    type: str = Field(..., description="Name of the regularization function used in training.")

    @field_validator("type")
    def validate_regular_function(cls, value):
        if value not in ACCEPTABLE_REGULARIZERS:
            print(f"Warning: '{value}' is not in the predefined list but will be allowed.")  
        return value

class CostFunctionDetails(BaseModel):
    lossFunction: LossFunctionDetails
    regularFunction: RegularFunctionDetails

class ObjectiveFunctionDetails(BaseModel):
    cost_function: CostFunctionDetails
    objectiveFunction: Literal[
        "minimize",
        "maximize"
    ] = Field(..., description="The objective of the loss function. The loss function is either minimized or maximized.")


class ObjectiveFunctionResponse(LLMResponse[ObjectiveFunctionDetails]):
    pass

"""Define Process Task Characterization"""
class TaskCharacterizationDetails(BaseModel):
    task_type: Literal[
        "Classification",
        "Adversarial",
        "Clustering",
        "Discrimination",
        "Generation",
        "Clustering",
        "Regression",
        "Unknown"
    ] = Field(..., description="The type of task that the model is being trained to perform.")

class TaskCharacterizationResponse(LLMResponse[TaskCharacterizationDetails]):
    pass
