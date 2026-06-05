# Gradino

Gradino is a multi-table non-relational numerical QA generator. It generates new QA instances  

1. from scratch  
2. for every user-defined domain (and in different languages, if needed)  
3. with predefined tabular perturbations (schema-level, cell-level, numerical-level)  
    a. layout-level: such as pivots, column merging etc.  
    b. cell-level: such as missing values, typographical noise etc.  
    c. numerical-level: such as unit of measurement heterogeneity  
4. with the option to generate parallel (i.e. evidence retrieved from each table, and then aggregated) and sequential (i.e. multi-hop questions across multiple tables) data.  
5. with total control over the number of tables, attributes, and rows per table.  

The Gradino-generated data described in the paper (available in `gradino.pdf`) is available in the `datasets/` directory. The finetuned models will be provided as HuggingFace models upon paper acceptance (due to the anonymity constraints).  

## How to generate new Table QA instances

Run the following command  

```bash
python3 main.py --domain [domain_name] --question_type [method_name] --num_tables [number_of_tables] --num_samples [number_of_instances_to_generate] --col_cardinality [number_elements_per_attributes] --num_columns [number_of_columns] [--sequential]
```

The supported domains are `environmental`, `finance`, `healthcare` and `products`, as the table generation procedure uses few-shot examples of each category to improve the nomenclature used in the tables.  
To add another domain, you can either add new few-shot tables in the newly created domain category in `prompts/example_tables`, or you can modify the generation procedure not to include few-shot examples.  
The `--col_cardinality` and `--num_columns` parameter control the generation of the latent relational source (see paper), not the final generated tables. For that, you would need to change the `num_columns_range` and `num_rows_range` variables in `perturbations.py`, as well as the view extraction procedure in the `get_table_view` functions in `models.py`. In general, it is better to keep these two parameters with their default values.  
For parallel instances, the `--col_cardinality` parameter is overridden so that the attribute cardinality is always equal to the number of tables requested (so that the parallel generation can succeed).  
The `--question_type`s available are `sum`, `average`, `superlative` (max, min).  

## Run ablation experiments

To run the ablation experiments, make sure not to pass the `--num_tables` parameter.

### Scaling and unit heterogeneity collapse

Run the following commands to generate ablations for the parallel and sequential data instances, as done in the main paper experiments (detailed in Table 4 and 5). Across each ablation, each scenario re-uses the same data, e.g. the 3-table scenario re-uses the same tables appearing in the 2-table scenario, and so on.  
For the parallel data instances, run the following command:

```bash
python3 main.py --domain environmental
```

and for sequential data,  

```bash
python3 main.py --domain environmental --sequential
```

This will by default generate the 2,3,5,10 and 20 table data for parallel (for both single- and mixed-unit scenarios), and 2,3 and 5 table data for sequential, inside the `datasets` directory. For each group of generated tables, Gradino automatically generates `sum`, `avg`, `superlative` (max, min) questions.  

### Running ablations on real tables

Run the following commands to replicate the results shown in Figure 8.  
To extract the tables:  

```bash
python3 table_downloader.py
```

To generate samples from those tables:  

```bash
python3 generate_real_ablations.py [--multitable]
```

To replicate the results shown in Figure 7, instead, run the following command, and modify the threshold variable inside the script:  

```bash
python3 extract_benchmark_tables.py
```

## Testing the models

To run inference on the data, use the `test.py` script.  

```bash
python3 test.py --filepath [path_to_dataset_csv] --model [model_name] [--tatqa | --real | --griqa] [--pot]
```

The options `--tatqa`, `--real` and `--griqa` are needed to correctly store the results. The `--pot` option will run the test with Program-of-Thoughts prompting (Table 8 in the paper).  
This script will save the results into the `results` directory, and will also generate the `reasoning.csv` files (reasoning traces) needed for finetuning.  

## Evaluating the models

To evaluate the results, use the `extract_metrics.py` script for Gradino-generated data.  

```bash
python3 extract_metrics.py --filepath [path_to_results_csv]
```

To evaluate on GRI-QA, run its specific evaluation script:  

```bash
python3 extract_metrics_griqa.py --filepath [path_to_results_csv]
```

## Finetuning on Gradino

We provide the finetuning script in `qwen_finetuning.py`. To run it, first make sure to adapt the OUTPUT_DIR (save location of the trained model) and base_path variables (f-string pointing to multiple `reasoning.csv` files, needed for training on the reasoning traces of bigger models).  
Also, make sure that the `reasoning.csv` files exist: run inference on the data first with `test.py`, that script will generate the `reasoning.csv` files in the `results` directory.

## How to customize Gradino for your needs

Gradino can be customized by  
1. Adding new domains: add few-shot examples of tables in the `prompts/example_tables` directory, and modify the `utils.py` to accept the newly created domain in the `--domain` parameter.  
2. Adding new question templates: now Gradino aggregates the data by using `sum`, `average` and `superlative` (max, min) templates, but new templates can be added by modifying the `SQL_TEMPLATES` variable in `models.py`.  
3. Adding new perturbations: new perturbations can be created by adding new functions in `perturbations.py`, and then adding the name of the function either inside the `pre_hct_perturbations` or `post_hct_perturbations` lists in the same file, depending on whether the perturbation should be applied before or after the pivot operation.  
    a. make sure to keep the same inputs and outputs across the different functions belonging to the same list, otherwise the `models.py` file needs to be modified as well.  
4. Changing the size of the tables: the size of the tables can be changed by modifying the `num_columns_range` and `num_rows_range` lists inside `perturbations.py`.  

For any doubts or implementation questions, please reach out to us by email or open a Github issue.  
