import argparse
import os
from typing import Iterator, Text, Tuple, Text as Path
import numpy as np
import tensorflow as tf
import shutil
import spacy
import anafora
import anafora.evaluate

# list of possible time expression labels
labels = [
    None,
    'AMPM-Of-Day',
    'After',
    'Before',
    'Between',
    'Calendar-Interval',
    'Day-Of-Month',
    'Day-Of-Week',
    'Frequency',
    'Hour-Of-Day',
    'Last',
    'Minute-Of-Hour',
    'Modifier',
    'Month-Of-Year',
    'Next',
    'NthFromStart',
    'Number',
    'Part-Of-Day',
    'Part-Of-Week',
    'Period',
    'Season-Of-Year',
    'Second-Of-Minute',
    'Sum',
    'This',
    'Time-Zone',
    'Two-Digit-Year',
    'Union',
    'Year',
]

# mapping from labels to integer indices
label_to_index = {l: i for i, l in enumerate(labels)}

# use spacy for tokenization, but make it split on a few more tokens
nlp = spacy.load("en_core_web_md")
infixes = [":", "/", "-"] + nlp.Defaults.infixes
infix_regex = spacy.util.compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_regex.finditer


def train(model_dir: Path,
          data_dir: Path,
          epochs: int,
          batch_size: int,
          learning_rate: float) -> None:
    debug = True

    # collect model inputs and outputs from training files
    inputs = []
    outputs = []
    for _, text, xml_path in iter_files(data_dir, "gold"):

        # convert text to token features and token labels
        token_offsets, token_indexes = text_to_token_offsets_and_indexes(text)
        token_labels = xml_to_token_labels(xml_path, token_offsets)
        inputs.append(token_indexes)
        outputs.append(token_labels)

        # for debugging purposes, print out the first document's labels
        if debug:
            print_token_info(text, token_offsets, token_indexes, token_labels)
            debug = False

    # flatten the inputs and outputs to predict each token independently
    #inputs = np.vstack([x.reshape(-1, 1) for x in inputs])
    #outputs = np.vstack([x.reshape(-1, 1) for x in outputs])
    max_token_length = max(sent.shape[1] for sent in inputs)

    total_sent_num = sum(sent.shape[0] for sent in inputs)
    input_matrix = np.zeros(shape=(total_sent_num, max_token_length))
    output_matrix = np.zeros(shape=(total_sent_num, max_token_length))

    sent_index = 0
    for k, text in enumerate(inputs):
        for i, sent in enumerate(text):
            for j, token in enumerate(sent):
                input_matrix[sent_index + i, j] = token
                output_matrix[sent_index + i, j] = outputs[k][i][j]
        sent_index = sent_index + i

    # create a model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(
            input_dim=len(nlp.vocab.vectors) + 1,
            output_dim=150),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.1, return_sequences=True)),
        #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, kernel_regularizer=tf.keras.regularizers.L2(0.01),
        #                                                   recurrent_regularizer=tf.keras.regularizers.l2(0.01),
        #                                                   bias_regularizer=tf.keras.regularizers.l2(0.01),
        #                                                   return_sequences=True)),
        #tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=len(labels), activation='softmax'),
    ])
    model.summary()
    # train the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.fit(input_matrix, output_matrix, epochs=epochs, batch_size=2)
    # save the model
    model.save(model_dir)


def predict(model_dir: Path,
            output_dir: Path,
            text_dir: Path,
            reference_dir: Path) -> None:
    debug = True

    # read the model
    model = tf.keras.models.load_model(model_dir)

    # write one file of predictions for each input text file
    for text_path, text, xml_path in iter_files(text_dir, "system"):

        # convert text to token features
        token_offsets, token_indexes = text_to_token_offsets_and_indexes(text)

        # get predictions from model
        orig_shape = token_indexes.shape
        predictions = model.predict(token_indexes)
        token_labels = np.argmax(predictions, axis=-1)

        # for debugging purposes, print out the first document's labels
        if debug:
            print_token_info(text, token_offsets, token_indexes, token_labels)
            debug = False

        # determine where to write the output XML
        output_path = os.path.join(output_dir,
                                   os.path.relpath(xml_path, text_dir))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # convert predictions into XML
        token_labels_to_xml(output_path, text_path, token_offsets, token_labels)

    # package the predictions up for CodaLab
    shutil.make_archive(output_dir, "zip", output_dir)

    # if a reference directory is provided, evaluate the predictions
    if reference_dir is not None:
        file_scores = anafora.evaluate.score_dirs(
            reference_dir, output_dir, exclude={("*", "<span>")})
        anafora.evaluate._print_merged_scores(
            file_scores, anafora.evaluate.Scores)


def text_to_token_offsets_and_indexes(text: Text)\
        -> Tuple[np.ndarray, np.ndarray]:

    # use Spacy to convert text into sentences and tokens
    sentences = list(nlp(text).sents)

    # create matrices of token offsets and indices
    max_tokens = max(len(s) for s in sentences)
    shape = (len(sentences), max_tokens)
    token_offsets = np.zeros(shape=shape + (2,), dtype=np.int32)
    token_indexes = np.zeros(shape=shape, dtype=np.int32)

    # fill matrices from the tokenized text
    for i, sentence in enumerate(sentences):
        for j, token in enumerate(sentence):

            # get token offsets from Spacy
            start = token.idx
            end = start + len(token.text)
            token_offsets[i, j] = [start, end]

            # get token index from Spacy
            token_index = token.rank + 1 if not token.is_oov else 0
            token_indexes[i, j] = token_index

    return token_offsets, token_indexes


def print_token_info(text: Text,
                     token_offsets: np.ndarray,
                     token_indexes: np.ndarray,
                     token_labels: np.ndarray) -> None:

    # iterate over sentences and tokens
    for i in range(token_labels.shape[0]):
        for j in range(token_labels.shape[1]):

            # only print non-None labels
            label_index = token_labels[i, j]
            label = labels[label_index]
            if label is not None:

                # print debugging info for this token
                start, end = token_offsets[i, j]
                print(f"{start:4n}:{end:<4n}"
                      f"  [{token_indexes[i, j]:5n}]->{label_index:2n}"
                      f"  '{text[start:end]}'->{label}'")


def iter_files(root_dir: Path, xml_type: Text)\
        -> Iterator[Tuple[Path, Text, Path]]:

    # walk down to each directory with online files (no subdirectories)
    for dir_path, dir_names, file_names in os.walk(root_dir):
        if not dir_names:

            k = '.DS_Store'
            while (k in file_names):
                file_names.remove(k)

            # read the text from the text file
            [text_file_name] = [f for f in file_names if not f.endswith(".xml")]
            text_path = os.path.join(dir_path, text_file_name)
            with open(text_path) as text_file:
                text = text_file.read()

            # calculate the XML file name from the text file name
            xml_path = f"{text_path}.TimeNorm.{xml_type}.completed.xml"

            # generate a tuple for this document
            yield text_path, text, xml_path


def xml_to_token_labels(xml_path: Path,
                        token_offsets: np.ndarray) -> np.ndarray:

    # read the XML file into an Anafora object
    data = anafora.AnaforaData.from_file(xml_path)

    # convert the XML information into labels for each character
    offset_labels = {}
    for annotation in data.annotations:
        for start, end in annotation.spans:
            for i in range(start, end):
                offset_labels[i] = annotation.type

    # assign each token the label of its first character
    token_labels = np.zeros(token_offsets.shape[:-1], dtype=np.int8)
    for i in range(token_labels.shape[0]):
        for j in range(token_labels.shape[1]):
            start, end = token_offsets[i, j]
            token_labels[i, j] = label_to_index[offset_labels.get(start)]
    return token_labels


def token_labels_to_xml(xml_path: Path,
                        text_path: Path,
                        token_offsets: np.ndarray,
                        token_labels: np.ndarray) -> None:

    # create an Anafora object to store the XML
    data = anafora.AnaforaData()
    entity_count = 0

    # iterate over sentences
    for i in range(token_offsets.shape[0]):
        entity = None

        # iterate over tokens in a sentence
        for j, (start, end) in enumerate(token_offsets[i]):

            # only create XML entities for non-padding, non-None predictions
            label_index = token_labels[i, j]
            if start != end and label_index != 0:

                # if the previous entity has a different label, create a new one
                if entity is None or label_index != token_labels[i, j - 1]:
                    entity = anafora.AnaforaEntity()
                    entity.id = f"{entity_count}@e@{text_path}@system"
                    entity.type = labels[label_index]
                    entity.spans = (start, end),
                    data.annotations.append(entity)
                    entity_count += 1

                # otherwise, update the previous entity's span
                else:
                    (old_start, _), = entity.spans
                    entity.spans = (old_start, end),

    # write the XML file
    data.to_file(xml_path)


if __name__ == "__main__":
    # sets up a command-line interface for "train" and "predict"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.set_defaults(func=train)
    train_parser.add_argument("model_dir")
    train_parser.add_argument("data_dir")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.set_defaults(func=predict)
    predict_parser.add_argument("model_dir")
    predict_parser.add_argument("output_dir")
    predict_parser.add_argument("text_dir")
    predict_parser.add_argument("--evaluate", dest='reference_dir')
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs.pop("func")(**kwargs)
