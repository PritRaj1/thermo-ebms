import grain.python as grain
import tensorflow_datasets as tfds
import numpy as np
from ml_collections import ConfigDict
from typing import Any


def test_data():
	return [
		{"image": np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)}
		for _ in range(50)
	]


def get_dataloader(name: str, split: str, batch_size: int) -> grain.IterDataset:
	"""Helper to build a single grain dataset."""
	is_training = split == "train"
	source = test_data() if name == "fake32" else tfds.data_source(name, split=split)

	ds = grain.MapDataset.source(source)

	if is_training:
		ds = ds.shuffle(seed=42)

	def preprocess(sample: dict[str, Any]) -> dict[str, Any]:
		image = sample["image"].astype(np.float32)
		image = (image / 127.5) - 1.0
		return {"x": image}

	ds = ds.map(preprocess)
	ds = ds.batch(batch_size=batch_size, drop_remainder=is_training)
	return ds.to_iter_dataset()


def get_loaders(data_config: ConfigDict) -> tuple[grain.IterDataset, grain.IterDataset]:
	name = data_config.dataset
	batch_size = data_config.batch_size
	train_loader = get_dataloader(name, "train", batch_size)
	test_loader = get_dataloader(name, "test", batch_size)

	num_examples = 500
	if name != "fake32":
		bldr = tfds.builder(name)
		num_examples = bldr.info.splits["train"].num_examples

	return train_loader, test_loader, num_examples // batch_size
