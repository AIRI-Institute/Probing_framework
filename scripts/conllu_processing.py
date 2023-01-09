import fire
import os
from typing import Optional
from probing.ud_parser.ud_parser import ConlluUDParser


def main(
    tr_path: Optional[os.PathLike] = None,
    va_path: Optional[os.PathLike] = None,
    te_path: Optional[os.PathLike] = None,
    dir_conllu_path: Optional[os.PathLike] = None,
    language: Optional[str] = None,
    save_path_dir: Optional[os.PathLike] = None,
    shuffle: bool = True,
    verbose: bool = True,
) -> None:
    converter = ConlluUDParser(shuffle, verbose)
    converter.convert(
        tr_path, va_path, te_path, dir_conllu_path, language, save_path_dir
    )


if __name__ == "__main__":
    fire.Fire(main)
