# How to use UD Parser

The conversion works in the following way:

**Data:** CONLLU files or a directory to such files for one language
**Result:** a file in SentEval format

read files;
find all morphological categories;
**foreach** *categories* **do**
> **foreach** *sentences* **do**
**if** *category is in sentence* **then**
get a category value
**end**
>
stratified split on three samples;
write to a file
**end**

UD Parser should be initialised with the following code:

```python
from probing.ud_parser.ud_parser import ConlluUDParser

splitter = ConlluUDParser()
```

UD Parser can be provided with the directory with CONLLU files or directly with paths to files:
```python
# You can provide a direct path to the folder with conllu files

splitter.convert(dir_conllu_path=<folder path>)

# Or you can pass paths to each of three possible conllu files

splitter.convert(tr_path=..., va_path=..., te_path=...)
```

There are several ways to generate files with different sorting mechanisms:
1. Only by morphological categories (the default option)
2. By parts of speech and morphological categories:
    ```python
    splitter = ConlluUDParser(sorting="by_pos")
    ```
3. By dependency relations and morphological categories
    ```python
    splitter = ConlluUDParser(sorting="by_deprel")
    ```
4. By parts of speech, dependency relations and morphological categories:
    ```python
    splitter = ConlluUDParser(sorting="by_pos_and_deprel")
    ```

