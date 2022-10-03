
�Lroot"_tf_keras_network*�L{"name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "dram_cpu_architecture"}, "name": "dram_cpu_architecture", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dram_container_memory_working_set_bytes"}, "name": "dram_container_memory_working_set_bytes", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dram_cache_miss"}, "name": "dram_cache_miss", "inbound_nodes": []}, {"class_name": "StringLookup", "config": {"name": "string_lookup_1", "trainable": true, "dtype": "int64", "invert": false, "max_tokens": null, "num_oov_indices": 0, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": [], "idf_weights": null, "encoding": "utf-8"}, "name": "string_lookup_1", "inbound_nodes": [[["dram_cpu_architecture", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_container_memory_working_set_bytes", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "name": "normalization_container_memory_working_set_bytes", "inbound_nodes": [[["dram_container_memory_working_set_bytes", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization_cache_miss", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "name": "normalization_cache_miss", "inbound_nodes": [[["dram_cache_miss", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_1", "trainable": true, "dtype": "float32", "num_tokens": 8, "output_mode": "one_hot", "sparse": false}, "name": "category_encoding_1", "inbound_nodes": [[["string_lookup_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization_container_memory_working_set_bytes", 0, 0, {}], ["normalization_cache_miss", 0, 0, {}], ["category_encoding_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dram_linear_regression_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dram_linear_regression_layer", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}], "input_layers": [["dram_container_memory_working_set_bytes", 0, 0], ["dram_cache_miss", 0, 0], ["dram_cpu_architecture", 0, 0]], "output_layers": [["dram_linear_regression_layer", 0, 0]]}, "shared_object_id": 11, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "dram_container_memory_working_set_bytes"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "dram_cache_miss"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "dram_cpu_architecture"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "dram_container_memory_working_set_bytes"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "dram_cache_miss"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "string", "dram_cpu_architecture"]}], "keras_version": "2.9.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "dram_cpu_architecture"}, "name": "dram_cpu_architecture", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dram_container_memory_working_set_bytes"}, "name": "dram_container_memory_working_set_bytes", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dram_cache_miss"}, "name": "dram_cache_miss", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "StringLookup", "config": {"name": "string_lookup_1", "trainable": true, "dtype": "int64", "invert": false, "max_tokens": null, "num_oov_indices": 0, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": true}, "name": "string_lookup_1", "inbound_nodes": [[["dram_cpu_architecture", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Normalization", "config": {"name": "normalization_container_memory_working_set_bytes", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "name": "normalization_container_memory_working_set_bytes", "inbound_nodes": [[["dram_container_memory_working_set_bytes", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Normalization", "config": {"name": "normalization_cache_miss", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "name": "normalization_cache_miss", "inbound_nodes": [[["dram_cache_miss", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_1", "trainable": true, "dtype": "float32", "num_tokens": 8, "output_mode": "one_hot", "sparse": false}, "name": "category_encoding_1", "inbound_nodes": [[["string_lookup_1", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization_container_memory_working_set_bytes", 0, 0, {}], ["normalization_cache_miss", 0, 0, {}], ["category_encoding_1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "dram_linear_regression_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dram_linear_regression_layer", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 10}], "input_layers": [["dram_container_memory_working_set_bytes", 0, 0], ["dram_cache_miss", 0, 0], ["dram_cpu_architecture", 0, 0]], "output_layers": [["dram_linear_regression_layer", 0, 0]]}}, "training_config": {"loss": "mean_absolute_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "coeff_determination", "dtype": "float32", "fn": "coeff_determination"}, "shared_object_id": 15}, {"class_name": "RootMeanSquaredError", "config": {"name": "root_mean_squared_error", "dtype": "float32"}, "shared_object_id": 16}, {"class_name": "MeanAbsoluteError", "config": {"name": "mean_absolute_error", "dtype": "float32"}, "shared_object_id": 17}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.5, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "dram_cpu_architecture", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "dram_cpu_architecture"}}2
�root.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "dram_container_memory_working_set_bytes", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dram_container_memory_working_set_bytes"}}2
�root.layer-2"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "dram_cache_miss", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dram_cache_miss"}}2
�root.layer-3"_tf_keras_layer*�{"name": "string_lookup_1", "trainable": true, "expects_training_arg": false, "dtype": "int64", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "StringLookup", "config": {"name": "string_lookup_1", "trainable": true, "dtype": "int64", "invert": false, "max_tokens": null, "num_oov_indices": 0, "oov_token": "[UNK]", "mask_token": null, "output_mode": "int", "sparse": false, "pad_to_max_tokens": false, "vocabulary": null, "idf_weights": null, "encoding": "utf-8", "has_input_vocabulary": true}, "inbound_nodes": [[["dram_cpu_architecture", 0, 0, {}]]], "shared_object_id": 3}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "normalization_container_memory_working_set_bytes", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_container_memory_working_set_bytes", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "inbound_nodes": [[["dram_container_memory_working_set_bytes", 0, 0, {}]]], "shared_object_id": 4, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "normalization_cache_miss", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "stateful": false, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization_cache_miss", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": []}, "mean": null, "variance": null}, "inbound_nodes": [[["dram_cache_miss", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}2
�root.layer-6"_tf_keras_layer*�{"name": "category_encoding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "CategoryEncoding", "config": {"name": "category_encoding_1", "trainable": true, "dtype": "float32", "num_tokens": 8, "output_mode": "one_hot", "sparse": false}, "inbound_nodes": [[["string_lookup_1", 0, 0, {}]]], "shared_object_id": 6}2
�root.layer-7"_tf_keras_layer*�{"name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["normalization_container_memory_working_set_bytes", 0, 0, {}], ["normalization_cache_miss", 0, 0, {}], ["category_encoding_1", 0, 0, {}]]], "shared_object_id": 7, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 8]}]}2
�	root.layer_with_weights-2"_tf_keras_layer*�{"name": "dram_linear_regression_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dram_linear_regression_layer", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["concatenate_1", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}2
�troot.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 19}2
�uroot.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "coeff_determination", "dtype": "float32", "config": {"name": "coeff_determination", "dtype": "float32", "fn": "coeff_determination"}, "shared_object_id": 15}2
�vroot.keras_api.metrics.2"_tf_keras_metric*�{"class_name": "RootMeanSquaredError", "name": "root_mean_squared_error", "dtype": "float32", "config": {"name": "root_mean_squared_error", "dtype": "float32"}, "shared_object_id": 16}2
�wroot.keras_api.metrics.3"_tf_keras_metric*�{"class_name": "MeanAbsoluteError", "name": "mean_absolute_error", "dtype": "float32", "config": {"name": "mean_absolute_error", "dtype": "float32"}, "shared_object_id": 17}2