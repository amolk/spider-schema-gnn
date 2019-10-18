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
        utterance = instance.fields['world'].metadata.db_context.utterance

        outputs = self._model.forward_on_instance(instance)

        json_output = {"utterance": utterance, "instances": []}
        unique_instances = {}
        for candidate in outputs['candidates']:
            discriminator_instance = candidate
            # discriminator_instance["utterance"] = utterance
            unique_key = str(discriminator_instance['tables_used'])  + \
                         str(discriminator_instance['columns_used']) + \
                         str(discriminator_instance['target'])
            unique_instances[unique_key] = discriminator_instance

        json_output["instances"] = list(unique_instances.values()) # unique by tables_used and columns_used

        return sanitize(json_output)

    # @overrides
    # def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
    #     """
    #     If you don't want your outputs in JSON-lines format
    #     you can override this function to output them differently.
    #     """
    #     lines = []
    #     for instance in outputs["instances"]:
    #         lines.append(str(instance))
    #         lines.append("\n")
    #     return ''.join(lines)
