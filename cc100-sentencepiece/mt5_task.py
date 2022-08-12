import functools

import seqio
import tensorflow as tf
import t5.data
from datasets import load_from_disk
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
from seqio import FunctionDataSource, utils

TaskRegistry = seqio.TaskRegistry
vocabulary = seqio.SentencePieceVocabulary("/home/stephen/Desktop/t5_test_run/t5x/HI_ALL_VOCAB_32000_UNIGRAM.model", extra_ids=100)


DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}



def gen_dataset(split, shuffle=False, seed=None, column="text", dataset_path=None):
    dataset = load_from_disk(dataset_path)
    if shuffle:
        if seed:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()
    while True:
        for item in dataset[str(split)]:
            yield item[column]


def dataset_fn(split, shuffle_files, seed=None, dataset_path=None):
    return tf.data.Dataset.from_generator(
        functools.partial(gen_dataset, split, shuffle_files, seed, dataset_path=dataset_path),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string, name=dataset_path)
    )


@utils.map_over_dataset
def target_to_key(x, key_map, target_key):
    """Assign the value from the dataset to target_key in key_map"""
    return {**key_map, target_key: x}

# link to the mt5 sentencepiece tokenizer vocabulary
vocabulary = seqio.SentencePieceVocabulary("/home/stephen/Desktop/t5_test_run/t5x/HI_ALL_VOCAB_32000_UNIGRAM.model", extra_ids=100)

#assamese_span_curruption
TaskRegistry.add(
    "assamese_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/ASSAMESE'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#bengali_span_curruption
TaskRegistry.add(
    "bengali_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/BENGALI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#bhisnupuriya_span_curruption
TaskRegistry.add(
    "bhisnupuriya_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/BHISNUPURIYA'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#bodo_span_curruption
TaskRegistry.add(
    "bodo_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/BODO'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#divehi_span_curruption
TaskRegistry.add(
    "divehi_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/DIVEHI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#dogri_span_curruption
TaskRegistry.add(
    "dogri_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/DOGRI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#english_span_curruption
TaskRegistry.add(
    "english_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/ENGLISH'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#gujarati_span_curruption
TaskRegistry.add(
    "gujarati_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/GUJARATI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#hindi_span_curruption
TaskRegistry.add(
    "hindi_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/HINDI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#kannada_span_curruption
TaskRegistry.add(
    "kannada_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/KANNADA'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#kashmiri_span_curruption
TaskRegistry.add(
    "kashmiri_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/KASHMIRI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#konkani_span_curruption
TaskRegistry.add(
    "konkani_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/KONKANI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#maithili_span_curruption
TaskRegistry.add(
    "maithili_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/MAITHILI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#malayalam_span_curruption
TaskRegistry.add(
    "malayalam_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/MALAYALAM'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#manipuri_span_curruption
TaskRegistry.add(
    "manipuri_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/MANIPURI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#marathi_span_curruption
TaskRegistry.add(
    "marathi_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/MARATHI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#nepali_span_curruption
TaskRegistry.add(
    "nepali_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/NEPALI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#odia_span_curruption
TaskRegistry.add(
    "odia_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/ODIA'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#panjabi_span_curruption
TaskRegistry.add(
    "panjabi_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/PANJABI'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#sanskrit_span_curruption
TaskRegistry.add(
    "sanskrit_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/SANSKRIT'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#tamil_span_curruption
TaskRegistry.add(
    "tamil_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/TAMIL'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#telugu_span_curruption
TaskRegistry.add(
    "telugu_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/TELUGU'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
#urdu_span_curruption
TaskRegistry.add(
    "urdu_span_curruption",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_path='/home/stephen/Desktop/MEGA_CORPUS/COMBINED_CORPUS/URDU'),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=None,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)

# defining a mixture of languages. 

#seqio mixture 3.5 
seqio.MixtureRegistry.add(
  "mix_3.5",
  ["assamese_span_curruption", "bengali_span_curruption", 
  "bhisnupuriya_span_curruption", "bodo_span_curruption", 
  "divehi_span_curruption", "dogri_span_curruption", 
  "english_span_curruption", "gujarati_span_curruption",
  "hindi_span_curruption", "kannada_span_curruption", 
  "kashmiri_span_curruption", "konkani_span_curruption", 
  "maithili_span_curruption", "malayalam_span_curruption",
  "manipuri_span_curruption", "marathi_span_curruption",
  "nepali_span_curruption", "odia_span_curruption",
  "panjabi_span_curruption", "sanskrit_span_curruption",
  "tamil_span_curruption", "telugu_span_curruption",
   "urdu_span_curruption" ],
  default_rate=3.5
)

seqio.MixtureRegistry.add(
  "mix_3",
  ["assamese_span_curruption", "bengali_span_curruption", 
  "bhisnupuriya_span_curruption", "bodo_span_curruption", 
  "divehi_span_curruption", "dogri_span_curruption", 
  "english_span_curruption", "gujarati_span_curruption",
  "hindi_span_curruption", "kannada_span_curruption", 
  "kashmiri_span_curruption", "konkani_span_curruption", 
  "maithili_span_curruption", "malayalam_span_curruption",
  "manipuri_span_curruption", "marathi_span_curruption",
  "nepali_span_curruption", "odia_span_curruption",
  "panjabi_span_curruption", "sanskrit_span_curruption",
  "tamil_span_curruption", "telugu_span_curruption",
   "urdu_span_curruption" ],
  default_rate=3
)

seqio.MixtureRegistry.add(
  "mix_2",
  ["assamese_span_curruption", "bengali_span_curruption", 
  "bhisnupuriya_span_curruption", "bodo_span_curruption", 
  "divehi_span_curruption", "dogri_span_curruption", 
  "english_span_curruption", "gujarati_span_curruption",
  "hindi_span_curruption", "kannada_span_curruption", 
  "kashmiri_span_curruption", "konkani_span_curruption", 
  "maithili_span_curruption", "malayalam_span_curruption",
  "manipuri_span_curruption", "marathi_span_curruption",
  "nepali_span_curruption", "odia_span_curruption",
  "panjabi_span_curruption", "sanskrit_span_curruption",
  "tamil_span_curruption", "telugu_span_curruption",
   "urdu_span_curruption" ],
  default_rate=2
)

seqio.MixtureRegistry.add(
  "mix_4",
  ["assamese_span_curruption", "bengali_span_curruption", 
  "bhisnupuriya_span_curruption", "bodo_span_curruption", 
  "divehi_span_curruption", "dogri_span_curruption", 
  "english_span_curruption", "gujarati_span_curruption",
  "hindi_span_curruption", "kannada_span_curruption", 
  "kashmiri_span_curruption", "konkani_span_curruption", 
  "maithili_span_curruption", "malayalam_span_curruption",
  "manipuri_span_curruption", "marathi_span_curruption",
  "nepali_span_curruption", "odia_span_curruption",
  "panjabi_span_curruption", "sanskrit_span_curruption",
  "tamil_span_curruption", "telugu_span_curruption",
   "urdu_span_curruption" ],
  default_rate=4
)