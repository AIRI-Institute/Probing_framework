import fire
import os
from typing import Optional
from probing.ud_parser.ud_parser import Splitter


def main(
    tr_path: Optional[os.PathLike] = None,
    va_path: Optional[os.PathLike] = None,
    te_path: Optional[os.PathLike] = None,
    dir_conllu_path: Optional[os.PathLike] = None,
    language: str = "",
    shuffle: bool = True,
    save_path_dir: Optional[os.PathLike] = None
) -> None:
    splitter = Splitter(language, shuffle, save_path_dir)
    _ = splitter.convert(tr_path, va_path, te_path, dir_conllu_path)


if __name__ == "__main__":
    fire.Fire(main)