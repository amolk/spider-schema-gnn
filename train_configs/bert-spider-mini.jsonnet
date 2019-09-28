local dataset_path = "./datasets/spider/";

{
  "random_seed": 5,
  "numpy_seed": 5,
  "pytorch_seed": 5,
  "dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "keep_if_unparsable": false,
    "loading_limit": -1,
    "question_token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased",
        "do_lowercase": true
      }
    }
  },
  "validation_dataset_reader": {
    "type": "spider",
    "tables_file": dataset_path + "tables.json",
    "dataset_path": dataset_path + "database",
    "lazy": false,
    "keep_if_unparsable": true,
    "loading_limit": -1
  },
  "train_data_path": dataset_path + "train_spider_mini.json",
  "validation_data_path": dataset_path + "train_spider_mini.json",
  "model": {
    "type": "spider",
    "dataset_path": dataset_path,
    "parse_sql_on_decoding": true,
    "gnn": true,
    "gnn_timesteps": 2,
    "decoder_self_attend": true,
    "decoder_use_graph_entities": true,
    "use_neighbor_similarity_for_linking": true,
    "question_embedder": {
      "allow_unmatched_keys": true,
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model": "bert-base-uncased"
      }
    },
    "action_embedding_dim": 768,
    "encoder": {
      "type": "lstm",
      "input_size": 1536,
      "hidden_size": 768,
      "bidirectional": true,
      "num_layers": 1
    },
    "entity_encoder": {
      "type": "boe",
      "embedding_dim": 768,
      "averaged": true
    },
    "decoder_beam_search": {
      "beam_size": 10
    },
    "training_beam_size": 1,
    "max_decoding_steps": 100,
    "input_attention": {"type": "dot_product"},
    "past_attention": {"type": "dot_product"},
    "dropout": 0.5
  },
  "iterator": {
    "type": "basic",
    "batch_size" : 10
  },
  "validation_iterator": {
    "type": "basic",
    "batch_size" : 1
  },
  "trainer": {
    "num_epochs": 100,
    "cuda_device": -1,
    "patience": 20,
    "validation_metric": "+sql_match",
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "num_serialized_models_to_keep": 2
  }
}
