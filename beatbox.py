import datasets
import csv
import os

# For a future citation perhaps?
# _CITATION = """\
# @inproceedings{luong-vu-2016-non,
#     title = "A non-expert {K}aldi recipe for {V}ietnamese Speech Recognition System",
#     author = "Luong, Hieu-Thi  and
#       Vu, Hai-Quan",
#     booktitle = "Proceedings of the Third International Workshop on Worldwide Language Service Infrastructure and Second Workshop on Open Infrastructures and Analysis Frameworks for Human Language Technologies ({WLSI}/{OIAF}4{HLT}2016)",
#     month = dec,
#     year = "2016",
#     address = "Osaka, Japan",
#     publisher = "The COLING 2016 Organizing Committee",
#     url = "https://aclanthology.org/W16-5207",
#     pages = "51--55",
# }
# """

_DESCRIPTION = """\
    Dataset consisting of isolated beatbox samples ,
    reimplementation of the dataset from the following 
    paper: BaDumTss: Multi-task Learning for Beatbox Transcription
"""

_HOMEPAGE = "https://doi.org/10.1007/978-3-031-05981-0_14"

_LICENSE = "MIT"

_DATA_URL = "https://huggingface.co/datasets/maxardito/beatbox/resolve/main/dataset"


class BeatboxDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            features=datasets.Features({
                "path":
                datasets.Value("string"),
                "class":
                datasets.Value("string"),
                "audio":
                datasets.Audio(sampling_rate=16_000),
            }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            # citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_manager.download_config.ignore_url_params = True

        audio_path = dl_manager.download(_DATA_URL)
        local_extracted_archive = dl_manager.extract(
            audio_path) if not dl_manager.is_streaming else None
        path_to_clips = "dataset"

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "local_extracted_archive":
                    local_extracted_archive,
                    "audio_files":
                    dl_manager.iter_archive(audio_path),
                    "metadata_path":
                    dl_manager.download_and_extract(
                        "dataset/metadata_train.csv.gz"),
                    "path_to_clips":
                    path_to_clips,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "local_extracted_archive":
                    local_extracted_archive,
                    "audio_files":
                    dl_manager.iter_archive(audio_path),
                    "metadata_path":
                    dl_manager.download_and_extract(
                        "dataset/metadata_test.csv.gz"),
                    "path_to_clips":
                    path_to_clips,
                },
            ),
        ]

    def _generate_examples(
        self,
        local_extracted_archive,
        audio_files,
        metadata_path,
        path_to_clips,
    ):
        """Yields examples."""
        data_fields = list(self._info().features.keys())
        metadata = {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["path"] = os.path.join(path_to_clips, row["path"])
                # if data is incomplete, fill with empty values
                for field in data_fields:
                    if field not in row:
                        row[field] = ""
                metadata[row["path"]] = row
        id_ = 0
        for path, f in audio_files:
            if path in metadata:
                result = dict(metadata[path])
                # set the audio feature and the path to the extracted file
                path = os.path.join(local_extracted_archive,
                                    path) if local_extracted_archive else path
                result["audio"] = {"path": path, "bytes": f.read()}
                result["path"] = path
                yield id_, result
                id_ += 1
