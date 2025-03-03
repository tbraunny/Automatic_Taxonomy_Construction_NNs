from typing import TypeVar, Generic
from pydantic.generics import GenericModel
from pydantic import BaseModel, Field
from typing import Literal, Optional


# Define a generic response model for structured LLM output.
T = TypeVar("T")
class LLMResponse(GenericModel, Generic[T]):
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
class OptimizerDetails(BaseModel):
    optimizer: Literal[
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
    learning_rate: float = Field(..., description="The learning rate used by the optimizer.")
    momentum: float = Field(..., description="The momentum value used by the optimizer.")

# Now define a specialized response type using the generic interface.
class TrainingOptimizerResponse(LLMResponse[OptimizerDetails]):
    pass
