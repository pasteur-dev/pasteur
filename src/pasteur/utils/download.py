import os
import subprocess
from .data import RawSource as DS

import logging

logger = logging.getLogger(__name__)


def download_files(name: str, dir: str, files: list[str]):
    if not files:
        assert False, "Empty file list"

    logger.info(f"Downloading dataset {name} files iteratively with wget.")
    args = ["wget", "-m", "-np", "-nH", "-c", "-P", dir]

    template_fn = files[0]
    # We have to skip parent dirs manually
    cut_dirs = len(template_fn.split("/")) - 4
    if cut_dirs > 0:
        args.append(f"--cut-dirs={cut_dirs}")

    args.extend(files)
    subprocess.run(args)


def download_index(
    name: str, download_dir: str, url_dir: str, username: str | None = None
):
    logger.info(f"Downloading dataset {name} through its index listing and wget.")
    assert url_dir[-1] == "/", "Url dir should end with a `/`"

    args = ["wget", "-m", "-np", "-nH", "-c", "-P", download_dir]

    # We have to skip parent dirs manually
    cut_dirs = len(url_dir.split("/")) - 4
    if cut_dirs > 0:
        args.append(f"--cut-dirs={cut_dirs}")

    args.append(url_dir)
    if username:
        args.extend(["--user", username, "--ask-password"])
    subprocess.run(args)


def download_s3(name: str, download_dir: str, bucket: str):
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config
    except Exception:
        assert False, "Specified dataset requires the aws package 'boto3'"

    logger.info(f"Downloading dataset {name} from s3 using boto3.")
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    ds_bucket = s3.Bucket(bucket) # type: ignore

    for s3_object in ds_bucket.objects.all():
        _, filename = os.path.split(s3_object.key)
        fn = os.path.join(download_dir, filename)
        if os.path.isfile(fn):
            logger.info(f"File already downloaded, skipping: {filename}")
            continue

        logger.info(f"Downloading {filename} ({s3_object.size / 1e6:.3f} mb)")
        ds_bucket.download_file(s3_object.key, fn)


def main(download_dir: str, datasets: dict[str, DS], username: str | None):
    assert os.path.exists(
        download_dir
    ), f'Download path "{download_dir}" doesn\'t exist.'

    for name, ds in datasets.items():
        save_name = ds.save_name or name
        save_path = os.path.join(download_dir, save_name)
        os.makedirs(save_path, exist_ok=True)

        if ds.credentials:
            assert username, f"Dataset requires credentials, use --user <user>"

        if isinstance(ds.files, list):
            download_files(name, save_path, ds.files)
        else:
            assert isinstance(ds.files, str)
            if ds.files.startswith("s3:"):
                download_s3(name, save_path, ds.files.replace("s3:", ""))
            else:
                download_index(
                    name, save_path, ds.files, username if ds.credentials else None
                )


def get_description(datasets: dict[str, DS]):
    desc = "The following data stores are available:\n"
    for name, ds in datasets.items():
        desc += f"{name:15s}: {ds.desc or ''}\n"
    return desc
