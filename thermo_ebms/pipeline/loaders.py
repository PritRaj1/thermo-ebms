import jax
import numpy as np
from PIL import Image
import grain.python as grain
import tensorflow_datasets as tfds
from typing import Any

from ..config import TrainingConfig


def test_data():
	return [
		{"image": np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)}
		for _ in range(50)
	]


class PreprocessTransform(grain.MapTransform):
	def __init__(self, image_res: int | None = None):
		self.image_res = image_res

	def map(self, sample: dict[str, Any]) -> dict[str, Any]:
		image = sample["image"]

		if self.image_res is not None:
			image = np.array(
				Image.fromarray(image).resize((self.image_res, self.image_res))
			)

		image = (image.astype(np.float32) / 127.5) - 1.0
		return {"x": image}


def get_dataloader(
	name: str, split: str, batch_size: int, seed: int, image_res: int | None = None
) -> grain.DataLoader:
	"""Helper to build a single grain dataset using the official DataLoader API."""
	is_training = split == "train"
	source = test_data() if name == "fake32" else tfds.data_source(name, split=split)

	sampler = grain.IndexSampler(
		num_records=len(source),
		num_epochs=None,
		shard_options=grain.ShardOptions(
			shard_index=jax.process_index(),
			shard_count=jax.process_count(),
			drop_remainder=True,
		),
		shuffle=is_training,
		seed=seed,
	)

	operations = [
		PreprocessTransform(image_res=image_res),
		grain.Batch(batch_size=batch_size, drop_remainder=True),
	]

	data_loader = grain.DataLoader(
		data_source=source,
		sampler=sampler,
		operations=operations,
		worker_count=0,
	)

	return data_loader


def get_loaders(
	data_config: TrainingConfig, seed: int
) -> tuple[grain.IterDataset, grain.IterDataset]:
	name = data_config.dataset
	batch_size = data_config.global_batch_size // jax.process_count()
	image_res = data_config.image_res
	train_loader = get_dataloader(name, "train", batch_size, seed, image_res=image_res)

	num_examples = 50
	if name != "fake32":
		bldr = tfds.builder(name)
		num_examples = bldr.info.splits["train"].num_examples

	return train_loader, num_examples // data_config.global_batch_size
