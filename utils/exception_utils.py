class AppError(Exception):
    def __init__(self, message, code=None, context=None):
        self.message = message
        self.code = code
        self.context = context or {}
        super().__init__(message)

    def to_dict(self):
        return {
            "error": self.message,
            "code": self.code,
            "context": self.context
        }

raise AppError(
    message="Could not parse ontology due to unsupported datatype.",
    code="UNSUPPORTED_DATATYPE",
    context={"datatype": "421", "property": "someDataProperty"}
)

try:
    delete_ann_configuration(ontology, "GAN")
except AppError as e:
    print("Error occurred:", e.to_dict())

try:
    result = delete_ann_configuration(ontology, "GAN")
    return {"success": True, "result": result}
except AppError as e:
    return {"success": False, **e.to_dict()}

Use Subclasses for Specific Error Types

For extra clarity and more targeted exception handling:

class OntologyParseError(AppError): pass
class ANNNotFoundError(AppError): pass