# import pandas as pd
# import tensorflow as tf
# import os

# import tensorflow.keras.datasets.imdb as IMDB
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import (
#     Embedding,
#     Dense,
#     LSTM,
# )
# from tensorflow.keras.preprocessing import sequence
# from sklearn.datasets import fetch_california_housing
# from readml.explainers.dl.explain_dl import ExplainDL
# from readml.logger import ROOT_DIR


# tf.compat.v1.disable_v2_behavior()


# def initialize_directories_dl(out_path, dir_to_create):
#     os.chdir(ROOT_DIR)
#     new_root = os.getcwd()
#     new_root = "/".join(new_root.split("/")[:-1])
#     os.chdir(new_root)
#     start = out_path.index("/") + 1
#     split = out_path[start:].split("/")
#     for elt in split:
#         if not os.path.isdir(elt):
#             os.makedirs(elt)
#             os.chdir(elt)
#         else:
#             os.chdir(elt)
#     os.chdir(ROOT_DIR)

#     for elt in dir_to_create:
#         if not os.path.isdir(os.path.join(out_path, elt)):
#             os.makedirs(os.path.join(out_path, elt))


# def create_dir_tabular_text():
#     dir_to_create = ["tabular", "text"]
#     out_path = "../outputs/tests/dl"
#     initialize_directories_dl(out_path, dir_to_create)


# create_dir_tabular_text()
# output_path_tabular_dir = os.path.join(ROOT_DIR, "../outputs/tests/dl", "tabular")
# output_path_text_dir = os.path.join(ROOT_DIR, "../outputs/tests/dl", "text")


# def test_explain_text():
#     if os.listdir(output_path_text_dir) != []:
#         for files in os.listdir(output_path_text_dir):
#             os.remove(os.path.join(output_path_text_dir, files))

#     dl_explain_text(output_path_text_dir)
#     min_obs, max_obs = 1, 5
#     output_path_local_min_obs = os.path.join(
#         output_path_text_dir, f"text_deep_local_explanation_{min_obs}th_obs.html"
#     )
#     output_path_local_max_obs = os.path.join(
#         output_path_text_dir, f"text_deep_local_explanation_{max_obs}th_obs.html"
#     )
#     outside_output = os.path.join(
#         output_path_text_dir, f"text_deep_local_explanation_{max_obs + 1}th_obs.html"
#     )

#     assert os.path.isfile(output_path_local_min_obs)
#     assert os.path.isfile(output_path_local_max_obs)
#     assert not os.path.isfile(outside_output)


# def dl_explain_text(output_path_text_dir):
#     X_train, y_train, max_words, word2idx = create_text_data()
#     max_review = 300
#     X_train = sequence.pad_sequences(X_train, maxlen=max_review)
#     train_data = pd.DataFrame(X_train)
#     model = simple_model_text(train_data, y_train, max_words, max_review)
#     train_data["target"] = y_train
#     exp = ExplainDL(
#         model=model,
#         out_path=output_path_text_dir,
#     )
#     exp.explain_text(
#         test_data=train_data.head(),
#         target_col="target",
#         word2idx=word2idx,
#     )


# def create_text_data():
#     max_word = 1000
#     (X_train, y_train), (X_test, y_test) = IMDB.load_data(num_words=max_word)
#     word_index = IMDB.get_word_index()
#     word2idx = {v: k for k, v in word_index.items()}
#     return X_train[0:10], y_train[0:10], max_word, word2idx


# def simple_model_text(X_train, y_train, max_words, max_review):
#     embedding_vector_length = 32
#     model = Sequential()
#     model.add(Embedding(max_words, embedding_vector_length, input_length=max_review))
#     model.add(LSTM(100))
#     model.add(Dense(1, activation="sigmoid"))
#     model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
#     model.fit(X_train, y_train)
#     return model


# def test_explain_tabular():
#     if os.listdir(output_path_tabular_dir) != []:
#         for files in os.listdir(output_path_tabular_dir):
#             os.remove(os.path.join(output_path_tabular_dir, files))

#     dl_explain_tabular(output_path_tabular_dir)
#     min_obs, max_obs = 0, 5
#     output_path_local_min_obs = os.path.join(
#         output_path_tabular_dir, f"tab_deep_local_explanation_{min_obs + 1}th_obs.html"
#     )
#     output_path_local_max_obs = os.path.join(
#         output_path_tabular_dir, f"tab_deep_local_explanation_{max_obs}th_obs.html"
#     )
#     outside_output = os.path.join(
#         output_path_tabular_dir, f"tab_deep_local_explanation_{max_obs + 1}th_obs.html"
#     )

#     assert os.path.isfile(output_path_local_min_obs)
#     assert os.path.isfile(output_path_local_max_obs)
#     assert not os.path.isfile(outside_output)


# def dl_explain_tabular(output_path):
#     X_train, y_train, data_train = create_tabular_data()
#     model = simple_model_regression(X_train, y_train)
#     exp = ExplainDL(
#         model=model,
#         out_path=output_path,
#     )
#     exp.explain_tabular(
#         data_train.head(),
#         features_name=X_train.columns.tolist(),
#         task_name="regression",
#     )


# def create_tabular_data():
#     dict_data = fetch_california_housing()
#     X_train = pd.DataFrame(dict_data["data"], columns=dict_data["feature_names"]).query(
#         "index < 10"
#     )
#     y_train = pd.DataFrame(
#         dict_data["target"], columns=["target"]
#     ).query("index < 10")
#     data_train = X_train.copy()
#     data_train[y_train.columns.values[0]] = y_train
#     return X_train, y_train, data_train


# def simple_model_regression(X_train, y_train):
#     input_dim = X_train.shape[1]
#     model = Sequential()
#     model.add(
#         Dense(13, input_dim=input_dim, kernel_initializer="normal", activation="relu")
#     )
#     model.add(Dense(1, kernel_initializer="normal"))
#     model.compile(loss="mean_squared_error", optimizer="adam")
#     model.fit(X_train, y_train, epochs=3, batch_size=32)
#     return model
