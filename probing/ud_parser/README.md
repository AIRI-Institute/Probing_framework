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
    
    
## Example
```python
from probing.ud_parser.ud_parser import ConlluUDParser

splitter = ConlluUDParser()
splitter.convert(tr_path="tests/parser_test/hi_pud-ud-test.conllu")
```

Output:
```
In progress:
tests/parser_test/hi_pud-ud-test.conllu
17 categories were found
Collecting data for no_sorting
Category "Foreign" has only one class
Not enough data of category "Mood" for stratified split
Category "NumType" has only one class
Category "Polarity" has only one class
Category "Polite" has only one class
Not enough data of category "PronType" for stratified split
Category "VerbForm" has only one class
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Animacy.csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Aspect.csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Case.csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Definite.csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Gender.csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Gender[psor].csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Number.csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Number[psor].csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Person.csv
Writing to file: /home/jovyan/Probing_probing/tests/parser_test/hi_pud_no_sorting_Tense.csv
```


