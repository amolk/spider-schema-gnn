import pdb
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor


@Predictor.register("spider_discriminator")
class DiscriminatorDatasetGenerator(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        json_output = {}
        outputs = self._model.forward_on_instance(instance)
        json_output = outputs

        return sanitize(json_output)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return str(outputs) + "\n"
