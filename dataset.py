import tensorflow_datasets as tfds

def getData():
    config = tfds.translate.opus.OpusConfig(
        version=tfds.core.Version('0.1.0'),
        language_pair=("en", "es"),
        subsets=["GNOME", "EMEA"]
    )

    builder = tfds.builder(name="opus", config=config)
    builder.download_and_prepare()

    print(builder.info)

    return builder.as_dataset(split='train', shuffle_files=True)