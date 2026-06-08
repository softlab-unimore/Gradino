import os
import re
import pickle
import numpy as np

from inference_models import OpenAIModel, extract_result, remove_markdown_syntax
from prompts.generate_relational_table import prompt as prompt_generate_relational_table
from prompts.generate_relational_table import prompt_domain as prompt_generate_relational_table_domain
from prompts.generate_relational_table import prompt_domain_exactly as prompt_generate_relational_table_domain_exactly
from prompts.generate_relational_table import prompt_domain_unit_specific as prompt_generate_relational_table_domain_unit
from prompts.generate_relational_table import prompt_domain_unit_specific_exactly as prompt_generate_relational_table_domain_unit_exactly
from prompts.generate_question_prompt import prompt as prompt_generate_question
from prompts.generate_question_prompt import prompt_percentage_change as prompt_generate_question_percentage_change
from prompts.generate_question_prompt import prompt_comparison as prompt_generate_question_comparison
from prompts.generate_question_prompt import prompt_multi as prompt_generate_question_multi
from prompts.generate_question_prompt import prompt_multi_unit_variation as prompt_generate_question_multi_unit_variation
from prompts.generate_question_prompt import prompt_multi_fk as prompt_generate_question_multi_fk
from prompts.constraint_prompt import constraint_prompt_gpt as prompt_constraint
from prompts.question_verification import prompt as prompt_verify_question
from prompts.question_verification import prompt_multi as prompt_verify_question_multi
from prompts.question_verification import prompt_multi_fk as prompt_verify_question_multi_fk
from prompts.question_verification import prompt_multi_unit_variation as prompt_verify_question_multi_unit_variation

from sql_sampler import SQLSampler
from perturbations import Perturber, IntraAmbiguousPerturber
from constrainer import Constrainer, parse_expr, extract_columns
from unit_converter import is_unit_in_domain, get_random_unit, get_value, get_n_canonical_units, is_value_going_up
import random
import pandas as pd
import itertools
import operator
import math
from tqdm import tqdm
from copy import deepcopy

from utils import is_float

SQL_TEMPLATES = {
    "extractive": """SELECT {target} FROM {table_name} WHERE {constraints};""",
    "comparative": """SELECT (SELECT {target} FROM {table_name} WHERE {constraints1}) {comparison} (SELECT {target} FROM {table_name} WHERE {constraints2}) as {target};""",
    "superlative": """SELECT {aggr}({target}) AS {target} FROM ({union_table});""",
    "sum": """SELECT {aggr}({target}) AS {target} FROM ({union_table});""",
    "average": """SELECT {aggr}({target}) AS {target} FROM ({union_table});""",
}

class MTAutoGen:
    def __init__(self, args):
        self.args = args
        self.gpt_model = OpenAIModel(temperature=0.7, p=0.8)
        self.num_columns_range = [4,5]
        self.past_table_names = []
        self.past_table_values = []

        self.sql_sampler = SQLSampler()
        self.perturber = Perturber()
        self.constrainer = Constrainer()
        self.intra_ambiguous_perturber = IntraAmbiguousPerturber()
        self.num_perturbations = 1
        self.rnd = random.Random(-1)

    def set_random_seed(self, i):
        self.rnd = random.Random(i)

    def check_table_sanity(
            self,
            result: dict,
            num_columns: int,
    ) -> bool:
        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, units, decimals = \
        result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
        result["value_col"], result["unit_of_measurement"], result["number_of_decimals"]

        sanity = True
        sanity = sanity and (num_columns == len(table_attributes))
        sanity = sanity and (len(table_attributes) == len(table_attributes_long) == len(table_types) == len(ranges))
        sanity = sanity and (value_col in table_attributes)
        sanity = sanity and value_col in table_attributes
        if not sanity:
            return sanity
        idx = table_attributes.index(value_col)

        units_cleaned = set([unit for unit in units if unit is not None and unit != "None"])
        sanity = sanity and (len(units_cleaned.difference(set(decimals.keys()))) == 0) #every unit of measurement must have a number of decimals defined

        sanity = sanity and (not " " in value_col) and (not "," in value_col)
        return sanity

    def generate_relational_table(
            self,
            domain: str = None,
            num_columns: int = None,
            col_cardinality: int = 10,
            canonical_units: list = []
    ):
        if num_columns is None:
            num_columns = random.randint(self.num_columns_range[0], self.num_columns_range[1])

        if domain is not None:
            examples = ""
            if domain in ["finance", "healthcare", "products", "environmental"]:
                path = f"prompts/example_tables/{domain}/"
                for filename in os.listdir(path):
                    if not filename.endswith(".html"):
                        continue

                    table = pd.read_html(os.path.join(path, filename))[0]
                    try:
                        table.columns = [col if "unnamed" not in col.lower() else "" for col in table.columns]
                    except:
                        pass

                    table = table.to_html(index=False) # removing useless style attributes
                    examples += table + "\n\n"
            else:
                raise NotImplementedError

            examples = examples.strip()
            attr = {
                "num_columns": num_columns,
                "col_cardinality": col_cardinality,
                "past": ", ".join(self.past_table_names),
                "past_values": ", ".join(self.past_table_names),
                "domain": domain,
                "examples": examples
            }

            if not canonical_units:
                result_query, _ = self.gpt_model.query(prompt_generate_relational_table_domain_exactly, attr=attr)
            else:
                attr["units"] = canonical_units
                result_query, _ = self.gpt_model.query(prompt_generate_relational_table_domain_unit_exactly, attr=attr)
        else:
            attr = {"num_columns": num_columns, "col_cardinality": col_cardinality, "past": ", ".join(self.past_table_names), "past_values": ", ".join(self.past_table_names)}
            result_query, _ = self.gpt_model.query(prompt_generate_relational_table, attr=attr)

        if result_query is not None:
            try:
                result = eval(remove_markdown_syntax(extract_result(result_query, "Final answer: ")))
            except:
                return None

            sanity = self.check_table_sanity(result, num_columns)
            if not sanity:
                print("wrong sanity")
                return None
        else:
            return None

        return result

    def generate_semantic_constraints(self, input, domain):
        attr = {"input": input, "domain": domain}
        result_query, _ = self.gpt_model.query(prompt_constraint, attr=attr)
        if result_query is not None:
            try:
                result = eval(remove_markdown_syntax(extract_result(result_query, "Final answer:")))
            except:
                return None
        else:
            return None

        return result

    def fill_dense_relational_table(
            self,
            table_attributes,
            table_types,
            ranges,
            units,
            decimals,
            value_col,
            pivot_keys=None,
            shuffle=False,
            semantic_constraints=None,
            col_to_keep=None,
            value_to_keep=None,
    ):
        if pivot_keys is None:
            pivot_keys = [attr for attr, typ in zip(table_attributes, table_types) if typ == "categorical" or (typ == "int" and attr != value_col)]

        attr_info = list(zip(table_attributes, table_types, ranges))

        type_value_col = table_types[table_attributes.index(value_col)]
        pos_value_col = table_attributes.index(value_col)
        pivot_info = [list(info) for info in attr_info if info[0] in pivot_keys]
        value_info = [list(info) for info in attr_info if info[0] == value_col and type_value_col in ["float", "int"]]

        # generate full cartesian product of pivot values
        if col_to_keep is not None and value_to_keep is not None:
            pos_col_to_keep = table_attributes.index(col_to_keep)
            type_col_to_keep = table_types[pos_col_to_keep]
            if type_col_to_keep == "int":
                domains_col_to_keep = [value_to_keep]
                domains_col_to_keep.extend([random.randint(*ranges[pos_col_to_keep]) for _ in range(4)])
                for idx in range(len(pivot_info)):
                    if pivot_info[idx][0] == col_to_keep:
                        pivot_info[idx][2] = domains_col_to_keep
                        break

        pivot_domains = [info[2] for info in pivot_info]
        pivot_attr_names = [info[0] for info in pivot_info]
        pivot_combinations = list(itertools.product(*pivot_domains))
        combinations = pivot_combinations

        rows = []
        for combo in combinations:
            row = list(combo)
            for attr, typ, rng in value_info:
                if typ == "int":
                    if is_float(rng[0]) and math.isinf(rng[0]):
                        if rng[0] < 0:
                            first_value = -4_000_000_000
                        else:
                            first_value = 4_000_000_000
                    else:
                        first_value = rng[0]

                    if is_float(rng[1]) and math.isinf(rng[1]):
                        if rng[1] < 0:
                            second_value = -4_000_000_000
                        else:
                            second_value = 4_000_000_000
                    else:
                        second_value = rng[1]
                    val = self.rnd.randint(int(first_value), int(second_value))
                elif typ == "float":
                    val = round(self.rnd.uniform(rng[0], rng[1]), int(decimals[units[0]]))
                elif typ == "categorical":
                    val = self.rnd.choice(rng)
                else:
                    raise ValueError(f"Unknown type {typ}")

                # insert the val at the correct pos_value_col position
                row.insert(pos_value_col, val)

            rows.append(row)

        final_columns = [info[0] for info in pivot_info]
        for attr, _, _ in value_info:
            final_columns.insert(table_attributes.index(attr), attr)

        df = pd.DataFrame(rows, columns=final_columns)
        df = df.drop_duplicates(subset=pivot_attr_names, keep="first")

        if semantic_constraints is not None and isinstance(semantic_constraints, dict) and "intra_row_constraints" in semantic_constraints and "inter_row_constraints" in semantic_constraints:
            df, df_constraints = self.constrainer.get_bounds(df, semantic_constraints, value_col, ranges, random_state=self.rnd.randint(0, 10**9))
            for i,_ in df.iterrows():
                if type_value_col == "float":
                    df.at[i, value_col] = round(df.at[i, value_col], decimals[units[0]])

        if df is not None and shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        return df

    def fill_relational_table(self, table_attributes: list, table_types: list, ranges: list):
        num_rows = self.rnd.randint(self.num_rows_range[0], self.num_rows_range[1])
        table = []
        for i in range(num_rows):
            row = []
            for j in range(len(table_attributes)):
                if table_types[j] == "categorical":
                    value = self.rnd.choice(ranges[j])
                elif table_types[j] == "float":
                    value = round(self.rnd.uniform(ranges[j][0], ranges[j][1]), 6)
                elif table_types[j] == "int":
                    value = self.rnd.randint(ranges[j][0], ranges[j][1])
                else:
                    raise ValueError(f"Unknown attribute type: {table_types[j]}")

                row.append(value)
            table.append(row)

        return pd.DataFrame(table, columns=table_attributes)

    def choose_constraint_on_new_table(self,
            columns,
            column_to_vary,
            old_col_values,
            table,
        ):

        colname = columns[column_to_vary]

        candidates = [v for v in table[colname].unique() if v not in old_col_values]

        if not candidates:
            return None, None

        vc = table[colname].value_counts(dropna=False)
        best_val = max(candidates, key=lambda v: (int(vc.get(v, 0)), str(v)))

        return best_val

    def generate_label(
            self, table: pd.DataFrame,
            table_name: str,
            value_col: str,
            method="extractive",
            columns: list = None,
            constraint: list = None,
            column_to_vary: int = None,
            old_col_values: list = None,
            best_val = None,
            impose_target_for_extractive: bool = False,
            all_cols: bool = False,
            col_to_keep: str = None,
            value_to_keep: str = None
    ):
        self.sql_sampler.clear_memory()

        apply_multi_table = False
        if (columns is not None or constraint is not None) and method != "extractive":
            # constrained label generation is only supported for the multi-table scenario, where "extractive" is used
            return None, None
        elif (columns is not None or constraint is not None) and method == "extractive" and len(old_col_values) > 0:
            apply_multi_table = True

        if method in ["extractive", "comparative", "superlative", "sum", "average", "percentage_change"]:
            if not apply_multi_table:
                clusters, columns = self.sql_sampler.get_combination(table, table_name, value_col, all_cols=all_cols,
                                                                     col_to_keep=col_to_keep, value_to_keep=value_to_keep)

                if len(clusters) == 0:
                    return None, None
            else:
                self.sql_sampler.load_table(table, table_name)

            if method == "extractive":
                if not apply_multi_table:
                    constraint = self.rnd.choice(clusters)

                if len(constraint) == len(table.columns):
                    return None, None

                columns_aval = [col for col in table.columns if col not in columns]
                if len(columns_aval) == 0:
                    return None, None

                if column_to_vary is not None and best_val is not None:
                    # in multi-table, we always get the numerical value
                    target = value_col

                    if isinstance(constraint, tuple):
                        constraint = list(constraint)
                    constraint[column_to_vary] = best_val
                else:
                    if impose_target_for_extractive:
                        target = value_col
                    else:
                        target = self.rnd.choice(columns_aval)

                constraint_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint)])
                formats = {"target": target, "constraints": constraint_txt, "table_name": table_name}
            elif method == "comparative":
                if len(clusters) < 2:
                    return None, None

                constraint = self.rnd.sample(clusters, 2)
                if any([len(c) == len(table.columns) for c in constraint]):
                    return None, None

                target = value_col
                constraint1_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint[0])])
                constraint2_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint[1])])
                comparison = self.rnd.choice([">", "<", "=", ">=", "<="])
                formats = {"target": target, "constraints1": constraint1_txt, "constraints2": constraint2_txt, "table_name": table_name, "comparison": comparison}
            elif method in ["superlative", "sum", "average"]:
                if method == "superlative":
                    num_constraint = self.rnd.choice([el for el in range(3,5)])
                else:
                    num_constraint = self.rnd.choice([el for el in range(2,5)])

                if len(clusters) < num_constraint:
                    return None, None

                constraint = self.rnd.sample(clusters, num_constraint)
                if any([len(c) == len(table.columns) for c in constraint]):
                    return None, None

                target = value_col
                if method == "superlative":
                    aggr = self.rnd.choice(["MAX", "MIN"])
                elif method == "sum":
                    aggr = "SUM"
                else:
                    aggr = "AVG"

                union_all_selects = []
                for constr_pair in constraint:
                    where_rules = []
                    for col, constr in zip(columns, constr_pair):
                        where_rules.append(f"{col} = \"{constr}\"")
                    where_rules = " AND ".join(where_rules)
                    union_all_selects.append(f"SELECT {target} AS {target} FROM {table_name} WHERE {where_rules}")

                union_table = " UNION ALL ".join(union_all_selects)
                formats = {"aggr": aggr, "union_table": union_table, "target": target}
            elif method == "percentage_change":
                if len(clusters) < 2:
                    return None, None

                constraint = self.rnd.sample(clusters, 2)
                if any([len(c) == len(table.columns) for c in constraint]):
                    return None, None

                target = value_col
                constraint1_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint[0])])
                constraint2_txt = " AND ".join([f"{col} = \"{constr}\"" for col, constr in zip(columns, constraint[1])])
                formats = {"target": target, "constraints1": constraint1_txt, "constraints2": constraint2_txt, "table_name": table_name}

            full_mask = pd.Series(False, index=table.index)
            if not isinstance(constraint[0], tuple):
                constraint_tmp = [constraint]
            else:
                constraint_tmp = constraint

            """
            constraint_tmp is basically always [constraint]
            we keep the for loop to facilitate further developments: technically, it is possible that we need to apply,
            for multiple values, multiple constraints.
            """
            for constrs in constraint_tmp:
                mask = pd.Series(True, index=table.index)
                for col, constr in zip(columns, constrs):
                    mask &= (table[col] == constr)
                full_mask |= mask

            sql_query = SQL_TEMPLATES[method].format(**formats)
            data = {
                "query": sql_query,
                "target": target,
                "constraint": constraint if not isinstance(constraint[0], tuple) else sum(constraint, ()),
                "columns": columns if method == "extractive" else columns*len(constraint)
            }
        else:
            raise NotImplementedError()

        result = self.sql_sampler.execute(sql_query)

        result = result[target].item()
        if method == "comparative":
            if result == 0:
                result = "no"
            else:
                result = "yes"

        if is_float(result):
            result = round(float(result), 6)

        data["label"] = str(result)
        return data, full_mask

    def substitute_column_names(self, table: pd.DataFrame, new_column_names: list[str]):
        assert len(new_column_names) == len(table.columns)
        mapping = {old: new for old, new in zip(table.columns, new_column_names)}
        table_renamed = table.rename(columns=mapping)
        return table_renamed

    def generate_label_multitable(self, list_of_data, method):
        data_merged = {
            "query": [],
            "target": [],
            "constraint": [],
            "columns": [],
            "labels": []
        }
        for data in list_of_data:
            data_merged["query"].append(data["query"])
            data_merged["target"].append(data["target"])
            data_merged["constraint"].extend(data["constraint"])
            data_merged["columns"].extend(data["columns"])
            data_merged["labels"].append(data["label"])

        data_merged["method"] = method

        if method == "comparative":
            if len(data_merged["labels"]) != 2:
                raise ValueError("comparative multi-table requires exactly 2 tables")
            if not is_float(data_merged["labels"][0]) or not is_float(data_merged["labels"][1]):
                return None

            ops = {
                ">": operator.gt,
                "<": operator.lt,
                "==": operator.eq,
                ">=": operator.ge,
                "<=": operator.le,
            }

            comparison = self.rnd.choice([">", "<", "==", ">=", "<="])
            data_merged["comparison"] = f"first_value {comparison} second_value"
            if ops[comparison](float(data_merged["labels"][0]), float(data_merged["labels"][1])):
                data_merged["label"] = "yes"
            else:
                data_merged["label"] = "no"
        elif method in ["superlative", "sum", "average"]:
            if False in [is_float(label) for label in data_merged["labels"]]: #not is_float(data_merged["labels"][0]) or not is_float(data_merged["labels"][1]):
                return None

            if method == "superlative":
                comparison = self.rnd.choice([max, min])
                data_merged["comparison"] = "get the maximum value" if comparison.__name__ == "max" else "get the minimum value"
            elif method == "sum":
                comparison = sum
                data_merged["comparison"] = "get the sum of the values"
            elif method == "average":
                comparison = lambda x: sum(x) / len(x)
                data_merged["comparison"] = "get the average of the values"

            data_merged["label"] = str(comparison([float(label) for label in data_merged["labels"]]))
        elif method == "percentage_change":
            if len(data_merged["labels"]) != 2:
                raise ValueError("comparative multi-table requires exactly 2 tables")
            if not is_float(data_merged["labels"][0]) or not is_float(data_merged["labels"][1]):
                return None
            if float(data_merged["labels"][1]) == 0:
                return None

            data_merged["comparison"] = "((first_value * second_value) * 100) / second_value"
            data_merged["label"] = str(float(data_merged["labels"][0]) - float(data_merged["labels"][1]) * 100.0 / float(data_merged["labels"][1]))

        return data_merged

    def generate_label_multitable_fk(self, list_of_data, method):
        data_merged = {
            "query": [],
            "target": [],
            "constraint": [],
            "columns": [],
            "labels": []
        }

        for data in list_of_data:
            data_merged["query"].append(data["query"])
            data_merged["target"].append(data["target"])
            data_merged["constraint"].extend(data["constraint"])
            data_merged["columns"].extend(data["columns"])
            data_merged["labels"].append(data["label"])

        data_merged["method"] = method
        data_merged["label"] = data_merged["labels"][-1]

        return data_merged

    def check_nlquestion_validity(
            self,
            nl_question: str,
            sql_query: str,
            table: str | list[str],
            label: str,
            multi: bool = False,
            unit: str = None,
    ):
        if isinstance(table, list):
            table = "\n\n".join([f"Table {i+1}:\n{t}" for i,t in enumerate(table)])
        attr = {"nl_question": nl_question, "table": table, "sql_question": sql_query, "sql_result": label}

        if unit is None:
            result, _ = self.gpt_model.query(prompt_verify_question if not multi else prompt_verify_question_multi, attr=attr)
        else:
            attr["unit"] = unit
            if not multi:
                raise ValueError("cannot have unit specified but no multi setting")
            result, _ = self.gpt_model.query(prompt_verify_question_multi_unit_variation, attr=attr)

        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final answer:"))
            if "yes" in result.lower():
                return nl_question
            else:
                return result

        return result

    def check_nlquestion_validity_fk(
            self,
            nl_question: str,
            data: dict,
            table: str | list[str],
            label: str,
            multi: bool = False,
            unit: str = None,
    ):
        if isinstance(table, list):
            table = "\n\n".join([f"Table {i+1}:\n{t}" for i,t in enumerate(table)])

        for j, sql_query in enumerate(data["query"][:-1]):
            attributes = self.get_select_attributes(sql_query)
            if len(attributes) != 1:
                raise ValueError("only single attribute supported in multi-table with foreign key scenario")

            for k, sql_query2 in enumerate(data["query"][j+1:]):
                for attribute in attributes:
                    if re.search(rf'\b{re.escape(attribute)}\b\s*(?:=|!=|<>|>=|<=|>|<)\s*(?:"[^"]*"|\'[^\']*\'|\S+)', sql_query2, re.IGNORECASE):
                        data["query"][j+1+k] = self.substitute_where_clause(sql_query2, attribute)

        for j, sql_query in enumerate(data["query"][1:]):
            data["query"][j+1] = self.strip_target_where(sql_query)

        attr = {"nl_question": nl_question, "table": table, "sql_question": [data["query"][0], data["query"][-1]], "sql_result": label, "unit": unit} # [data["query"][0], data["query"][-1]]

        result, _ = self.gpt_model.query(prompt_verify_question if not multi else prompt_verify_question_multi_fk, attr=attr)

        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final answer:"))
            if "yes" in result.lower():
                return nl_question
            else:
                return result

        return result

    def check_nlquestion_validity_multi(self, nl_question: str, query_sqls: list[str], list_of_tables: list[str], label: str, aggregation: str, number_of_decimals: int):
        with pd.option_context("display.float_format", lambda x: f"{x:.{number_of_decimals}f}"):
            table_htmls = [table.to_html(index=False) for table in list_of_tables]
        text = "\n\n".join([f"SQL Query {i + 1}: {sql_query}\nTable {i + 1}: {table}" for i, (sql_query, table) in
                            enumerate(zip(query_sqls, table_htmls))])

        attr = {"nl_question": nl_question, "text": text, "aggregation": aggregation, "result": label}

        result, _ = self.gpt_model.query(prompt_verify_question, attr=attr)
        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final answer:"))
            if "yes" in result.lower():
                return nl_question
            else:
                return result

        return result

    def generate_question(self, table: pd.DataFrame, data: dict, number_of_decimals: int, method: str = None):
        with pd.option_context("display.float_format", lambda x: f"{x:.{number_of_decimals}f}"):
            table_html = table.to_html(index=False)
        query_sql = data["query"]
        label = data["label"]
        attr = {"table": table_html, "query": query_sql, "result": label}
        if method == "percentage_change":
            prompt_tmp = prompt_generate_question_percentage_change
        elif method == "comparative":
            prompt_tmp = prompt_generate_question_comparison
        else:
            prompt_tmp = prompt_generate_question

        result, _ = self.gpt_model.query(prompt_tmp, attr=attr, create_question=True)
        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final question:"))
        return result

    def generate_question_multitable(
            self,
            list_of_tables: list,
            data: dict,
            number_of_decimals: list[int],
            method: str = "superlative",
            unit: str = None
    ):
        table_htmls = []
        for i, table in enumerate(list_of_tables):
            with pd.option_context("display.float_format", lambda x: f"{x:.{number_of_decimals[i]}f}"):
                table_htmls.append(table.to_html(index=False))

        query_sqls = data["query"]
        label = data["label"]
        text = "\n\n".join([f"SQL Query {i+1}: {sql_query}\nTable {i+1}: {table}" for i, (sql_query, table) in enumerate(zip(query_sqls, table_htmls))])

        attr = {"text": text, "aggregation": data["comparison"], "result": label, "method": method}
        if unit is None:
            result, _ = self.gpt_model.query(prompt_generate_question_multi, attr=attr, create_question=True)
        else:
            attr["unit"] = unit
            result, _ = self.gpt_model.query(prompt_generate_question_multi_unit_variation, attr=attr, create_question=True)

        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final question:"))
        return result, table_htmls

    def get_select_attributes(self, sql_query: str):
        select_part = re.search(r"select\s+(.*?)\s+from\b", sql_query, re.IGNORECASE | re.DOTALL).group(1)
        return [x.strip() for x in select_part.split(",")]

    def substitute_where_clause(self, sql_query: str, attribute: str):
        return re.sub(
            rf'(\b{re.escape(attribute)}\b\s*(?:=|!=|<>|>=|<=|>|<)\s*)(?:"[^"]*"|\'[^\']*\'|\S+)',
            r'\1"this value depends on the previous instruction"',
            sql_query,
            flags=re.IGNORECASE
        )

    def strip_target_where(self, sql):
        target = "this value depends on the previous instruction"

        while True:
            i = sql.find(target)
            if i == -1:
                return sql

            left_and = sql.rfind(" AND ", 0, i)
            left_where = sql.rfind(" WHERE ", 0, i)

            end = i + len(target)
            if left_and > left_where:
                start = left_and
            else:
                start = left_where+7
                end += 5

            sql = (sql[:start] + sql[end:]).strip()

    def generate_question_multitable_fk(
            self,
            list_of_tables: list,
            data: dict,
            number_of_decimals: list[int],
            method: str = "superlative",
    ):
        table_htmls = []
        for i, table in enumerate(list_of_tables):
            with pd.option_context("display.float_format", lambda x: f"{x:.{number_of_decimals[i]}f}"):
                #table_htmls = [table.to_html(index=False) for table in list_of_tables]
                table_htmls.append(table.to_html(index=False))

        for j, sql_query in enumerate(data["query"][:-1]):
            attributes = self.get_select_attributes(sql_query)
            if len(attributes) != 1:
                raise ValueError("only single attribute supported in multi-table with foreign key scenario")

            for k, sql_query2 in enumerate(data["query"][j+1:]):
                for attribute in attributes:
                    if re.search(rf'\b{re.escape(attribute)}\b\s*(?:=|!=|<>|>=|<=|>|<)\s*(?:"[^"]*"|\'[^\']*\'|\S+)', sql_query2, re.IGNORECASE):
                        data["query"][j+1+k] = self.substitute_where_clause(sql_query2, attribute)

        for j, sql_query in enumerate(data["query"][1:]):
            data["query"][j+1] = self.strip_target_where(sql_query)

        query_sqls = data["query"]
        label = data["label"]
        #text = "\n\n".join([f"SQL Query {i+1}: {sql_query}\nTable {i+1}: {table}" for i, (sql_query, table) in enumerate(zip(query_sqls, table_htmls))])
        text = "SQL Queries:\n\n"+"\n\n".join(f"Query {i+1}:{query}" for i, query in enumerate([query_sqls[0], query_sqls[-1]]))
        text += "\n\nTables:\n\n" + "\n\n".join(f"Table {i+1}:\n{table}" for i, table in enumerate(table_htmls))

        attr = {"text": text, "result": label, "method": method}
        result, _ = self.gpt_model.query(prompt_generate_question_multi_fk, attr=attr, create_question=True)

        if result is not None:
            result = remove_markdown_syntax(extract_result(result, "Final question:"))
        return result, table_htmls

    def get_table_view(
            self,
            attributes: list,
            attributes_long: list,
            attributes_types: list,
            ranges: list,
            value_col: str,
            columns: list = None,
    ):
        """
        returns a random selection of attributes from a table, while keeping value_col
        """

        value_idx = attributes.index(value_col)
        indices = [value_idx]
        if columns is not None:
            indices.extend([attributes.index(col) for col in columns])
        indices = list(set(indices))

        if columns is None:
            num_columns_to_add = self.rnd.randint( # initially we get random columns so that tables can slightly vary more
                5 - len(indices),
                5 - len(indices)
            )
            possible_columns_to_add = [i for i in range(len(attributes)) if i not in indices]
        else:
            num_columns_to_add = self.rnd.randint(1, 2)
            possible_columns_to_add = []

        if len(possible_columns_to_add) != 0:
            columns_to_add = self.rnd.sample(possible_columns_to_add,
                                             min(len(possible_columns_to_add), num_columns_to_add))
            indices.extend(
                columns_to_add)  # random.sample([i for i in range(len(attributes)) if i not in indices], num_attrs-len(indices)))
            indices = sorted(list(set(indices)))
        else:
            columns_to_add = []

        indices = [i for i in indices if i != value_idx] + [value_idx]
        ranges_tmp = deepcopy(ranges)
        """if columns is not None:
            for i in columns_to_add:  # we reduce the amount of labels for the added columns to reduce computational complexity, while varying more the generated tables
                ranges_tmp[i] = self.rnd.sample(ranges_tmp[i], min(len(ranges_tmp[i]), self.rnd.randint(2, 3)))"""

        get_view = lambda x: [x[i] for i in indices]
        attributes_view, attributes_long_view, attributes_types_view, range_view = (get_view(attributes),
                                                                                    get_view(attributes_long),
                                                                                    get_view(attributes_types),
                                                                                    get_view(ranges_tmp))

        return attributes_view, attributes_long_view, attributes_types_view, range_view  # , [attributes[col] for col in columns_to_add]

    def get_table_view_fk(
            self,
            attributes: list,
            attributes_long: list,
            attributes_types: list,
            ranges: list,
            value_col: str,
            col_to_keep: str,
            cols_to_avoid: list,
            is_final: bool = False,
    ):
        """
        returns a random selection of attributes from a table, while keeping value_col
        """

        if is_final:
            indices = [attributes.index(value_col)]
        else:
            indices = []

        if col_to_keep is not None:
            indices.append(attributes.index(col_to_keep))

        indices = list(set(indices))

        num_columns_to_add = self.rnd.randint(
            5 - len(indices),
            5 - len(indices)
        )
        possible_columns_to_add = [i for i in range(len(attributes)) if attributes[i] not in cols_to_avoid+[col_to_keep, value_col] and attributes_types[i] != "float"]

        if len(possible_columns_to_add) != 0:
            columns_to_add = self.rnd.sample(possible_columns_to_add, min(len(possible_columns_to_add), num_columns_to_add))
            indices.extend(columns_to_add)
            indices = sorted(list(set(indices)))
        else:
            columns_to_add = []

        ranges_tmp = deepcopy(ranges)

        get_view = lambda x: [x[i] for i in indices]
        attributes_view, attributes_long_view, attributes_types_view, range_view = (get_view(attributes),
                                                                                   get_view(attributes_long),
                                                                                   get_view(attributes_types),
                                                                                   get_view(ranges_tmp))

        return attributes_view, attributes_long_view, attributes_types_view, range_view #, [attributes[col] for col in columns_to_add]

    def run_one_table(
            self,
            domain=None,
            col_cardinality=10,
            method=None,
    ):
        """
        we use this function to generate dataset samples for multiple question types (extractive, comparative etc.) for the same tables,
        and for multiple perturbation types (to reduce API costs)
        """
        aggr = "first"
        sampled_canonical_units = get_n_canonical_units(domain, n=1)

        # GENERATION SCHEMA
        result = self.generate_relational_table(
            domain=domain,
            num_columns=6,
            col_cardinality=col_cardinality,
            canonical_units=sampled_canonical_units,
        )

        if result is None:
            return None, None, "table generation error (result is None)"

        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, units, decimals = \
            result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
                result["value_col"], result["unit_of_measurement"], result["number_of_decimals"]

        if table_name is None:
            return None, None, "table generation error (table is None)"
        if sum([el == "float" for el in table_types]) > 1:
            return None, None, "table generation error (too many floats)"
        if is_unit_in_domain(units[0], domain) is None:
            return None, None, "wrong unit generated"

        # GENERATION OF SEMANTIC CONSTRAINTS
        constraints = self.generate_semantic_constraints(result, domain=domain)
        if constraints is None:
            return None, None, "semantic constraints generation error"

        initial_offset = random.randint(0, 10000) # used to guarantee that different runs use different random perturbations
        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        datasets = {}

        if method is not None:
            methods = [method]
        else:
            methods = SQL_TEMPLATES

        for method in methods:
            for idx in range(len(range_view)):
                if table_types_view[idx] not in ["float", "int"]:
                    range_view[idx] = range_view[idx][:6]  # we reduce the combinations of values to avoid too much overhead

            # LATENT RELATIONAL SOURCE GENERATION
            try:
                view_constraints = {"inter_row_constraints": [], "intra_row_constraints": []}
                for constraint_type in constraints:
                    for rule in constraints[constraint_type]:
                        ok = True
                        if constraint_type == "inter_row_constraints":
                            conditions = [m.group("cond") for m in
                                          re.compile(self.constrainer.ATOM_EXTRACT, re.VERBOSE | re.DOTALL).finditer(
                                              rule)]
                        else:
                            m = self.constrainer.PATTERN_intra.match(rule)
                            if not m:
                                raise ValueError(f"Invalid intra rule: {rule!r}")
                            conditions = [m.group("condition").strip()]
                            expr = [m.group("expr").strip()]
                            conditions.extend(expr)

                        for cond in conditions:
                            cond = cond.strip()
                            if cond[0] in ["(", ")"]:
                                cond = cond[1:]
                            if cond[-1] in ["(", ")"]:
                                cond = cond[:-1]
                            columns_from_rule = extract_columns(parse_expr(cond))
                            for col in columns_from_rule:
                                if col not in table_attributes_view:
                                    ok = False
                                    break
                            if not ok:
                                break

                        if ok:
                            view_constraints[constraint_type].append(rule)

                table = self.fill_dense_relational_table(table_attributes_view, table_types_view, range_view, units,
                                                         decimals, value_col,
                                                         semantic_constraints=view_constraints)

            except Exception as e:
                continue

            # run sql loading and sql generation/execution
            try:
                data, full_mask = self.generate_label(
                    table,
                    table_name,
                    value_col,
                    method=method,
                )
            except Exception as e:
                continue

            if data is None:
                continue

            new_table = table.copy()
            new_full_mask = full_mask
            new_table_types = deepcopy(table_types)

            self.perturber.set_random_seed(initial_offset + i)
            if self.rnd.randint(0,1):  # apply null perturbation with 50% chance
                attrs = {
                    "table": new_table,
                    "constraints": list(zip(data["columns"], data["constraint"])),
                    "value_col": value_col,
                }

                new_table = self.perturber.null_perturbation(**attrs)

            try:
                new_table = self.substitute_column_names(new_table, table_attributes_long_view)
                new_value_col = table_attributes_long[table_attributes.index(value_col)]
            except Exception as e:
                continue

            # apply pre-pivot perturbations
            for perturbation in self.perturber.pre_hct_perturbations:
                if self.rnd.randint(0,1):
                    attrs = {
                        "table": new_table,
                        "value_col": new_value_col
                    }

                    new_table, new_value_col = perturbation(**attrs)

            try:
                # unit change
                new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col,
                                                                                     new_units,
                                                                                     unit_in_cell=unit_in_cell)
                table_before_pivot = new_table.copy()
                # pivot
                table_hct, rows_chosen, cols_chosen, option = self.perturber.multiheader_perturbation(new_table,
                                                                                                      new_value_col,
                                                                                                      aggr=aggr,
                                                                                                      unit_in_cell=unit_in_cell,
                                                                                                      full_mask=new_full_mask)
            except Exception as e:
                continue

            if table_hct is None:
                continue

            table_hct = self.perturber.restore_needed_cells_after_value_merge(
                table_before_pivot=table_before_pivot,
                pivot_table=table_hct,
                value_col=new_value_col,
                full_mask=full_mask,
                rows_chosen=rows_chosen,
                cols_chosen=cols_chosen,
            )

            # apply post pivot perturbations
            for perturbation in self.perturber.post_hct_perturbations:
                if self.rnd.randint(0,1):
                    attrs = {
                        "table": table_hct
                    }

                    table_hct = perturbation(**attrs)

            if "nl_question" not in data:
                data["nl_question"] = self.generate_question(table_hct, data, decimals[units[0]], method=method)
                if data["nl_question"] is None:
                    continue

                with pd.option_context("display.float_format", lambda x: f"{x:.{decimals[units[0]]}f}"):
                    data["nl_question"] = self.check_nlquestion_validity(data["nl_question"], data["query"],
                                                                         table_hct.to_html(index=False),
                                                                         data["label"])
                if data["nl_question"] is None:
                    continue

            datasets[method] = (table_hct, data, constraints)

        return datasets, decimals[units[0]], None





    def run_one_table_ablations(
            self,
            domain=None,
            method=None,
            col_cardinality=10,
    ):
        """
        we use this function to generate dataset samples for multiple question types (extractive, comparative etc.) for the same tables,
        and for multiple perturbation types (to reduce API costs)
        """
        aggr = "first"
        sampled_canonical_units = get_n_canonical_units(domain, n=1)

        # GENERATION SCHEMA
        result = self.generate_relational_table(
            domain=domain,
            num_columns=6,
            col_cardinality=col_cardinality,
            canonical_units=sampled_canonical_units,
        )

        if result is None:
            return None, None, "table generation error (result is None)"

        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, units, decimals = \
            result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
                result["value_col"], result["unit_of_measurement"], result["number_of_decimals"]

        if table_name is None:
            return None, None, "table generation error (table is None)"
        if sum([el == "float" for el in table_types]) > 1:
            return None, None, "table generation error (too many floats)"
        if is_unit_in_domain(units[0], domain) is None:
            return None, None, "wrong unit generated"

        # GENERATION OF SEMANTIC CONSTRAINTS
        constraints = self.generate_semantic_constraints(result, domain=domain)
        if constraints is None:
            return None, None, "semantic constraints generation error"

        initial_offset = random.randint(0, 10000) # used to guarantee that different runs use different random perturbations
        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        datasets = {}
        perturbations_to_apply = [self.perturber.null_perturbation] + \
                                 self.perturber.pre_hct_perturbations + \
                                 self.perturber.post_hct_perturbations

        if method is not None:
            method_list = [method]
        else:
            method_list = SQL_TEMPLATES

        for method in method_list:
            for idx in range(len(range_view)):
                if table_types_view[idx] not in ["float", "int"]:
                    range_view[idx] = range_view[idx][:6] # we reduce the combinations of values to avoid too much overhead

            # LATENT RELATIONAL SOURCE GENERATION
            try:
                view_constraints = {"inter_row_constraints": [], "intra_row_constraints": []}
                for constraint_type in constraints:
                    for rule in constraints[constraint_type]:
                        ok = True
                        if constraint_type == "inter_row_constraints":
                            conditions = [m.group("cond") for m in re.compile(self.constrainer.ATOM_EXTRACT, re.VERBOSE | re.DOTALL).finditer(rule)]
                        else:
                            m = self.constrainer.PATTERN_intra.match(rule)
                            if not m:
                                raise ValueError(f"Invalid intra rule: {rule!r}")
                            conditions = [m.group("condition").strip()]
                            expr = [m.group("expr").strip()]
                            conditions.extend(expr)

                        for cond in conditions:
                            cond = cond.strip()
                            if cond[0] in ["(", ")"]:
                                cond = cond[1:]
                            if cond[-1] in ["(", ")"]:
                                cond = cond[:-1]
                            columns_from_rule = extract_columns(parse_expr(cond))
                            for col in columns_from_rule:
                                if col not in table_attributes_view:
                                    ok = False
                                    break
                            if not ok:
                                break

                        if ok:
                            view_constraints[constraint_type].append(rule)

                table = self.fill_dense_relational_table(table_attributes_view, table_types_view, range_view, units, decimals, value_col,
                                                             semantic_constraints=view_constraints)

            except Exception as e:
                continue

            # run sql loading and sql generation/execution
            try:
                data, full_mask = self.generate_label(
                    table,
                    table_name,
                    value_col,
                    method=method,
                )
            except Exception as e:
                continue

            if data is None:
                continue

            for i, perturbation in enumerate(perturbations_to_apply):
                new_table = table.copy()
                new_full_mask = full_mask
                new_table_types = deepcopy(table_types)

                self.perturber.set_random_seed(initial_offset+i)
                if perturbation == "null_perturbation": # apply null perturbation with 50% chance
                    attrs = {
                        "table": new_table,
                        "constraints": list(zip(data["columns"], data["constraint"])),
                        "value_col": value_col,
                    }

                    new_table = self.perturber.null_perturbation(**attrs)

                try:
                    new_table = self.substitute_column_names(new_table, table_attributes_long_view)
                    new_value_col = table_attributes_long[table_attributes.index(value_col)]
                except Exception as e:
                    continue

                # apply pre-pivot perturbations
                if perturbation in self.perturber.pre_hct_perturbations:
                    attrs = {
                        "table": new_table,
                        "value_col": new_value_col
                    }

                    new_table, new_value_col = perturbation(**attrs)

                try:
                    # unit change
                    new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, new_units,
                                                                                         unit_in_cell=unit_in_cell)
                    table_before_pivot = new_table.copy()
                    # pivot
                    table_hct, rows_chosen, cols_chosen, option = self.perturber.multiheader_perturbation(new_table, new_value_col,
                                                                        aggr=aggr,
                                                                        unit_in_cell=unit_in_cell, full_mask=new_full_mask)
                except Exception as e:
                    continue

                if table_hct is None:
                    continue

                table_hct = self.perturber.restore_needed_cells_after_value_merge(
                    table_before_pivot=table_before_pivot,
                    pivot_table=table_hct,
                    value_col=new_value_col,
                    full_mask=full_mask,
                    rows_chosen=rows_chosen,
                    cols_chosen=cols_chosen,
                )

                # apply post pivot perturbations
                if perturbation in self.perturber.post_hct_perturbations:
                    attrs = {
                        "table": table_hct
                    }

                    table_hct = perturbation(**attrs)

                if "nl_question" not in data:
                    data["nl_question"] = self.generate_question(table_hct, data, decimals[units[0]], method=method)
                    if data["nl_question"] is None:
                        continue

                    with pd.option_context("display.float_format", lambda x: f"{x:.{decimals[units[0]]}f}"):
                        data["nl_question"] = self.check_nlquestion_validity(data["nl_question"], data["query"],
                                                                             table_hct.to_html(index=False), data["label"])
                    if data["nl_question"] is None:
                        continue

                if method not in datasets:
                    datasets[method] = {}
                datasets[method][perturbation.__name__] = (table_hct, data, constraints)

        return datasets, decimals[units[0]], None

    def reduce_sql_where(self, sql: str, table: pd.DataFrame, kept_table_full_mask: np.ndarray) -> str:
        def find_keyword_outside(s, keyword, start=0):
            kw = keyword.lower()
            in_quote = None
            depth = 0
            i = start
            while i <= len(s) - len(keyword):
                ch = s[i]
                if in_quote:
                    if ch == in_quote:
                        in_quote = None
                else:
                    if ch in "\"'":
                        in_quote = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    elif depth == 0 and s[i:i + len(keyword)].lower() == kw:
                        left_ok = i == 0 or not (s[i - 1].isalnum() or s[i - 1] == "_")
                        right_ok = i + len(keyword) == len(s) or not (
                                    s[i + len(keyword)].isalnum() or s[i + len(keyword)] == "_")
                        if left_ok and right_ok:
                            return i
                i += 1
            return -1

        def split_top_level_and(where_txt):
            parts, buf = [], []
            in_quote = None
            depth = 0
            i = 0
            while i < len(where_txt):
                ch = where_txt[i]
                if in_quote:
                    buf.append(ch)
                    if ch == in_quote:
                        in_quote = None
                else:
                    if ch in "\"'":
                        in_quote = ch
                        buf.append(ch)
                    elif ch == "(":
                        depth += 1
                        buf.append(ch)
                    elif ch == ")":
                        depth -= 1
                        buf.append(ch)
                    elif depth == 0 and where_txt[i:i + 3].upper() == "AND":
                        left_ok = i == 0 or where_txt[i - 1].isspace()
                        right_ok = i + 3 == len(where_txt) or where_txt[i + 3].isspace()
                        if left_ok and right_ok:
                            parts.append("".join(buf).strip())
                            buf = []
                            i += 3
                            continue
                        buf.append(ch)
                    else:
                        buf.append(ch)
                i += 1
            parts.append("".join(buf).strip())
            return [p for p in parts if p]

        def split_select_from_where(select_sql):
            i_select = find_keyword_outside(select_sql, "SELECT")
            i_from = find_keyword_outside(select_sql, "FROM", i_select + 6)
            i_where = find_keyword_outside(select_sql, "WHERE", i_from + 4)
            if min(i_select, i_from, i_where) == -1:
                raise ValueError(f"Could not parse SELECT/FROM/WHERE in: {select_sql}")
            return (
                select_sql[i_select + 6:i_from].strip(),
                select_sql[i_where + 5:].strip().rstrip(";"),
            )

        def extract_target_col(select_expr):
            s = select_expr.strip()
            i_as = find_keyword_outside(s, "AS")
            if i_as != -1:
                s = s[:i_as].strip()
            if s.endswith(")"):
                par = s.find("(")
                if par != -1:
                    s = s[par + 1:-1].strip()
            return s.strip().strip('"').strip("'")

        def parse_condition(cond):
            in_quote = None
            depth = 0
            for i, ch in enumerate(cond):
                if in_quote:
                    if ch == in_quote:
                        in_quote = None
                else:
                    if ch in "\"'":
                        in_quote = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    elif ch == "=" and depth == 0:
                        left = cond[:i].strip()
                        right = cond[i + 1:].strip()
                        if len(right) >= 2 and right[0] in "\"'" and right[-1] == right[0]:
                            return left, right[1:-1]
                        raise ValueError(f"Unsupported condition: {cond}")
            raise ValueError(f"Unsupported condition: {cond}")

        def row_mask(conds):
            m = pd.Series(True, index=table.index)
            for cond in conds:
                col, val = parse_condition(cond)
                m &= table[col].astype(str).eq(val)
            return m

        def reduce_one_select(select_sql):
            select_expr, where_txt = split_select_from_where(select_sql)
            conds = split_top_level_and(where_txt)
            target_col = extract_target_col(select_expr)
            target_col_pos = table.columns.get_loc(target_col)

            for r in range(1, len(conds) + 1):
                for sub in itertools.combinations(conds, r):
                    rm = row_mask(sub).to_numpy()
                    if kept_table_full_mask[rm, target_col_pos].sum() == 1:
                        i_where = find_keyword_outside(select_sql, "WHERE")
                        return select_sql[:i_where + 5] + " " + " AND ".join(sub)
            return select_sql

        def split_top_level_union_all(sql_txt):
            parts = []
            in_quote = None
            depth = 0
            i = 0
            last = 0
            token = "UNION ALL"
            while i < len(sql_txt):
                ch = sql_txt[i]
                if in_quote:
                    if ch == in_quote:
                        in_quote = None
                else:
                    if ch in "\"'":
                        in_quote = ch
                    elif ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    elif depth == 0 and sql_txt[i:i + len(token)].upper() == token:
                        parts.append(sql_txt[last:i].strip())
                        last = i + len(token)
                        i += len(token)
                        continue
                i += 1
            parts.append(sql_txt[last:].strip())
            return parts

        sql = sql.strip().rstrip(";")

        if find_keyword_outside(sql, "WHERE") != -1:
            return reduce_one_select(sql) + ";"

        i_from = find_keyword_outside(sql, "FROM")
        if i_from == -1:
            raise ValueError(f"Could not parse SQL: {sql}")

        prefix = sql[:i_from + 4]
        inner = sql[i_from + 4:].strip()

        if not (inner.startswith("(") and inner.endswith(")")):
            raise ValueError(f"Could not parse SQL: {sql}")

        inner = inner[1:-1].strip()
        parts = split_top_level_union_all(inner)
        reduced = [reduce_one_select(p) for p in parts]
        return f"{prefix} (" + " UNION ALL ".join(reduced) + ");"

    def run_multi_table_ablations(
            self,
            domain=None,
            num_columns=6,
            num_tables=None,
            method=None,
    ):
        """
        we use this function to generate dataset samples for multiple question types (extractive, comparative etc.) for the same tables,
        and for multiple perturbation types (to reduce API costs)
        """
        aggr = "first"
        sampled_canonical_units = get_n_canonical_units(domain, n=20)

        if num_tables is not None and num_tables != -1:
            num_tables_list = [num_tables]
        else:
            num_tables_list = [2, 3, 5, 10, 20]

        col_cardinality = num_tables_list[-1]

        # generation of relational table
        result = self.generate_relational_table(
            domain=domain,
            num_columns=num_columns,
            col_cardinality=col_cardinality,
            canonical_units=sampled_canonical_units,
        )

        if result is None:
            return None, None, "table generation error (result is None)"

        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, units, decimals = \
            result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
                result["value_col"], result["unit_of_measurement"], result["number_of_decimals"]

        if table_name is None:
            return None, None, "table generation error (table is None)"
        if sum([el == "float" for el in table_types]) > 1:
            return None, None, "table generation error (too many floats)"
        if is_unit_in_domain(units[0], domain) is None:
            return None, None, "wrong unit generated"

        constraints = self.generate_semantic_constraints(result, domain=domain)
        if constraints is None:
            return None, None, "semantic constraints generation error"

        columns_considered_to_add = [col for col in table_attributes if col not in str(constraints["inter_row_constraints"]) and col not in str(constraints["intra_row_constraints"])]

        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        datasets = {}
        tables_unit_converted, tables_not_unit_converted = {}, {}
        data_unit_converted, data_not_unit_converted = {}, {}

        view_constraints_tot_unit, view_constraints_tot_not_unit = {}, {}
        initial_offset = random.randint(0, 10000) # used to guarantee that different runs use different random perturbations

        for it, num_tables in enumerate(num_tables_list):
            tables_unit_converted[str(num_tables)] = []
            tables_not_unit_converted[str(num_tables)] = []
            data_unit_converted[str(num_tables)] = []
            data_not_unit_converted[str(num_tables)] = []
            view_constraints_tot_unit[str(num_tables)] = []
            view_constraints_tot_not_unit[str(num_tables)] = []
            datasets[str(num_tables)] = {}

        columns, constraint_cols, column_to_vary, old_col_values = None, None, None, []
        good = True
        for it, num_tables in enumerate(num_tables_list):
            print(f"Generating for {num_tables} tables...")

            if it == 0:
                previous_num_tables = 0
            else:
                previous_num_tables = num_tables_list[it-1]

            for i in range(previous_num_tables, num_tables):
                self.set_random_seed(initial_offset+i)
                table_attributes_view, table_attributes_long_view, table_types_view, range_view = self.get_table_view(
                    table_attributes,
                    table_attributes_long,
                    table_types,
                    ranges,
                    value_col,
                    columns=columns,
                )

                if column_to_vary is None:
                    column_to_vary = self.rnd.choice([idx for idx in range(len(table_attributes_view)) if table_types_view[idx] not in ["float", "int"]])

                for idx in range(len(range_view)):
                    if idx != column_to_vary and table_types_view[idx] not in ["float", "int"]:
                        range_view[idx] = range_view[idx][:6] # we reduce the combinations of values to avoid too much overhead


                # generate random table by filling the values randomly
                try:
                    view_constraints = {"inter_row_constraints": [], "intra_row_constraints": []}
                    for constraint_type in constraints:
                        for rule in constraints[constraint_type]:
                            ok = True
                            if constraint_type == "inter_row_constraints":
                                conditions = [m.group("cond") for m in re.compile(self.constrainer.ATOM_EXTRACT, re.VERBOSE | re.DOTALL).finditer(rule)]
                            else:
                                #conditions = [rule]
                                m = self.constrainer.PATTERN_intra.match(rule)
                                if not m:
                                    raise ValueError(f"Invalid intra rule: {rule!r}")
                                conditions = [m.group("condition").strip()]
                                expr = [m.group("expr").strip()]
                                conditions.extend(expr)

                            for cond in conditions:
                                cond = cond.strip()
                                if cond[0] in ["(", ")"]:
                                    cond = cond[1:]
                                if cond[-1] in ["(", ")"]:
                                    cond = cond[:-1]
                                columns_from_rule = extract_columns(parse_expr(cond))
                                for col in columns_from_rule:
                                    if col not in table_attributes_view:
                                        ok = False
                                        break
                                if not ok:
                                    break

                            if ok:
                                view_constraints[constraint_type].append(rule)

                    table = self.fill_dense_relational_table(table_attributes_view, table_types_view, range_view, units, decimals, value_col,
                                                                 semantic_constraints=view_constraints)

                    best_val = None
                    if i != 0:
                        best_val = self.choose_constraint_on_new_table(
                            columns,
                            column_to_vary,
                            old_col_values,
                            table,
                        )

                        if isinstance(constraint_cols, tuple):
                            constraint_cols = list(constraint_cols)

                        constraint_cols[column_to_vary] = best_val

                        # rows where the selected columns match the given values
                        mask = (table[columns] == constraint_cols).all(axis=1)

                        # keep all non-matching rows, and deduplicate matching rows
                        table = pd.concat([
                            table[~mask],
                            table[mask].drop_duplicates(subset=columns, keep="first")
                        ], ignore_index=True)

                except Exception as e:
                    good = False
                    break

                # run sql loading and sql generation/execution
                try:
                    data, full_mask = self.generate_label(
                        table,
                        table_name,
                        value_col,
                        method="extractive", # we force the method to be extractive, as the true method will be applied across tables later
                        columns=columns,
                        constraint=constraint_cols,
                        column_to_vary=column_to_vary,
                        old_col_values=old_col_values,
                        best_val=best_val,
                        all_cols=True,
                        impose_target_for_extractive=True
                    )
                except Exception as e:
                    good = False
                    break

                if data is None:
                    good = False
                    break

                if columns is None and constraint_cols is None:
                    columns = data["columns"]
                    constraint_cols = data["constraint"]

                    if column_to_vary is None:
                        # I make sure that inter row constraints are kept even when dealing with multiple tables
                        # for that, in different tables, I vary an attribute that is not part of the inter-row constraints (if any)
                        columns_to_consider = [col for col in columns if
                                               col not in str(view_constraints["inter_row_constraints"])]
                        # the column to vary must not be the value_col
                        columns_to_consider = [col for col in columns_to_consider if col != value_col]
                        # the column to vary must have at least num_tables different values, to ensure that we can vary it across tables
                        columns_to_consider = [col for col in columns_to_consider if table[col].nunique() >= num_tables]
                        if len(columns_to_consider) == 0:
                            good = False
                            break

                        max_val, best_col = -1, None
                        for col in columns_to_consider:
                            pos = columns.index(col)
                            num_samples = len(table[table[columns[pos]] == data["constraint"][pos]])
                            if num_samples > max_val:
                                max_val = num_samples
                                best_col = col

                        column_to_vary = columns.index(best_col)

                old_col_values.append(data["constraint"][column_to_vary])

                table = table[table[columns[column_to_vary]] == data["constraint"][column_to_vary]]

                rnd_apply_nan = self.rnd.randint(0,1)
                unit_in_cell = self.rnd.randint(0, 1)
                rnd_apply_pre = []
                for _ in self.perturber.pre_hct_perturbations:
                    rnd_apply_pre.append(self.rnd.randint(0,1))

                rnd_apply_post = []
                for _ in self.perturber.post_hct_perturbations:
                    rnd_apply_post.append(self.rnd.randint(0,1))

                # adding a sample with all perturbations randomly applied
                for apply_unit_conversions in [True, False]:
                    new_table = table.copy()
                    new_full_mask = full_mask
                    new_table_types = deepcopy(table_types)

                    self.perturber.set_random_seed(initial_offset+i)
                    if rnd_apply_nan == 1: # apply perturbation with 50% chance
                        attrs = {
                            "table": new_table,
                            "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                            "value_col": value_col,
                        }

                        new_table = self.perturber.null_perturbation(**attrs)

                    try:
                        new_table = self.substitute_column_names(new_table, table_attributes_long_view)
                        new_value_col = table_attributes_long[table_attributes.index(value_col)]
                    except Exception as e:
                        good = False
                        break

                    try:
                        new_units = units
                        new_decimals = decimals
                        if i != 0 and apply_unit_conversions:
                            needed_decimals = -1
                            for j in range(5):
                                surrogate_unit, target_unit = get_random_unit(units[0], domain, seed=initial_offset+i+j)
                                highest_needed_decimals = -1
                                for k,row in new_table.iterrows():
                                    try:
                                        value_converted, needed_decimals = get_value(
                                            surrogate_unit,
                                            target_unit,
                                            domain,
                                            row[new_value_col],
                                            self.perturber.nan_fill_str
                                        )
                                    except:
                                        break

                                    if needed_decimals is not None:
                                        if needed_decimals == -1:
                                            break

                                        new_table.at[k, new_value_col] = value_converted
                                        if needed_decimals > highest_needed_decimals:
                                            highest_needed_decimals = needed_decimals

                                if needed_decimals != -1:
                                    break

                            if needed_decimals == -1:
                                good = False
                                break

                            if surrogate_unit != units[0]:
                                target_unit = f"{target_unit} {units[0].replace('[', '').replace(']', '')}"

                            new_units = [target_unit]
                            new_decimals = {target_unit: max(decimals[units[0]], highest_needed_decimals)}
                    except Exception as e:
                        good = False
                        break

                    for num_pert, perturbation in enumerate(self.perturber.pre_hct_perturbations):
                        if rnd_apply_pre[num_pert] == 1 and i != 0:
                            attrs = {
                                "table": new_table,
                                "value_col": new_value_col
                            }

                            new_table, new_value_col = perturbation(**attrs)

                    try:
                        new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, new_units,
                                                                                             unit_in_cell=unit_in_cell)
                        table_before_pivot = new_table.copy()
                        table_hct, rows_chosen, cols_chosen, option = self.perturber.multiheader_perturbation(new_table, new_value_col,
                                                                            aggr=aggr,
                                                                            unit_in_cell=unit_in_cell, full_mask=new_full_mask)
                    except Exception as e:
                        good = False
                        break

                    if table_hct is None:
                        good = False
                        break

                    table_hct = self.perturber.restore_needed_cells_after_value_merge(
                        table_before_pivot=table_before_pivot,
                        pivot_table=table_hct,
                        value_col=new_value_col,
                        full_mask=full_mask,
                        rows_chosen=rows_chosen,
                        cols_chosen=cols_chosen,
                    )

                    for num_pert, perturbation in enumerate(self.perturber.post_hct_perturbations):
                        if perturbation == self.perturber.insert_blank_columns and option == 0: # may cancel unit of measurement
                            continue
                        if rnd_apply_post[num_pert] == 1:
                            attrs = {
                                "table": table_hct
                            }

                            table_hct = perturbation(**attrs)

                    data["decimals"] = new_decimals[new_units[0]]
                    data["units"] = new_units[0]
                    if apply_unit_conversions:
                        for num_tables2 in num_tables_list:
                            if num_tables2 < num_tables:
                                continue
                            tables_unit_converted[str(num_tables2)].append(table_hct.copy())
                            data_unit_converted[str(num_tables2)].append(deepcopy(data))
                            view_constraints_tot_unit[str(num_tables2)].append(view_constraints)
                    else:
                        for num_tables2 in num_tables_list:
                            if num_tables2 < num_tables:
                                continue
                            tables_not_unit_converted[str(num_tables2)].append(table_hct.copy())
                            data_not_unit_converted[str(num_tables2)].append(deepcopy(data))
                            view_constraints_tot_not_unit[str(num_tables2)].append(view_constraints)

            if not good:
                return datasets, decimals[units[0]], None

            if method is not None:
                method_list = [method]
            else:
                method_list = SQL_TEMPLATES

            for perturbation_name, list_of_table_hct, list_of_data, view_constr in zip(
                    ["unit_converted", "not_unit_converted"],
                    [tables_unit_converted[str(num_tables)], tables_not_unit_converted[str(num_tables)]],
                    [data_unit_converted[str(num_tables)], data_not_unit_converted[str(num_tables)]],
                    [view_constraints_tot_unit[str(num_tables)], view_constraints_tot_not_unit[str(num_tables)]]
            ):
                for method_name in method_list:
                    if method_name in ["extractive", "comparative", "percentage_change"]:
                        continue

                    if method_name not in datasets[str(num_tables)]:
                        datasets[str(num_tables)][method_name] = {}

                    data = self.generate_label_multitable(list_of_data, method=method_name)
                    if data is None:
                        return None, None, "error in generating multitable data"

                    data["nl_question"], tables_final = self.generate_question_multitable(
                        list_of_table_hct,
                        data,
                        [d["decimals"] for d in list_of_data],
                        method_name,
                        unit=list_of_data[0]["units"]
                    )
                    if data["nl_question"] is None:
                        return None, None, "error in generating nl question"

                    data["nl_question"] = self.check_nlquestion_validity(
                        data["nl_question"],
                        data["query"],
                        tables_final,
                        data["label"],
                        multi=True,
                        unit=list_of_data[0]["units"]
                    )
                    if data["nl_question"] is None:
                        return None, None, "error in checking nl question"

                    data["decimals"] = [d["decimals"] for d in list_of_data]
                    datasets[str(num_tables)][method_name][perturbation_name] = (list_of_table_hct, data, view_constr)

        return datasets, decimals[units[0]], None



    def run_multi_table_ablations_fk( # table split by foreign key
            self,
            domain=None,
            num_tables=None,
            col_cardinality=8,
            num_columns=21,
            method=None,
    ):
        """
        we use this function to generate dataset samples for multiple question types (extractive, comparative etc.) for the same tables,
        and for multiple perturbation types (to reduce API costs)
        """
        aggr = "first"
        sampled_canonical_units = get_n_canonical_units(domain, n=20)

        # generation of schema
        result = self.generate_relational_table(
            domain=domain,
            num_columns=num_columns, #21 columns, so get_table_view_fk chooses 5 new attributes on the first table, and 4 on the others, allowing maximum 5 tables. This constraint can be relaxed by allowing duplicate data across tables
            col_cardinality=col_cardinality,
            canonical_units=sampled_canonical_units,
        )

        if result is None:
            return None, None, "table generation error (result is None)"

        table_name, table_attributes, table_attributes_long, table_types, ranges, value_col, units, decimals = \
            result["name"], result["attributes"], result["attributes_long"], result["attribute_types"], result["range"], \
                result["value_col"], result["unit_of_measurement"], result["number_of_decimals"]

        ranges = [r[:5] for r in ranges]

        if table_name is None:
            return None, None, "table generation error (table is None)"
        if sum([el == "float" for el in table_types]) > 1:
            return None, None, "table generation error (too many floats)"
        if is_unit_in_domain(units[0], domain) is None:
            return None, None, "wrong unit generated"

        # generation of semantic constraints
        constraints = self.generate_semantic_constraints(result, domain=domain)

        if constraints is None:
            return None, None, "semantic constraints generation error"

        self.past_table_names.append(", ".join(table_attributes))
        self.past_table_values.append(", ".join([str(r) for r in ranges]))
        self.past_table_names = self.past_table_names[-5:]
        self.past_table_values = self.past_table_values[-5:]

        datasets = {}
        errors = []

        initial_offset = random.randint(0, 10000)
        if num_tables is not None:
            range_tables = [num_tables]
        else:
            range_tables = [2, 3, 5]

        for num_tables_idx, num_tables in enumerate(range_tables):
            print(f"Generating with {num_tables} tables...")
            datasets[str(num_tables)] = {}

            if method is not None:
                method_list = [method]
            else:
                method_list = SQL_TEMPLATES

            for method_name in method_list:
                if method_name in ["extractive", "comparative", "percentage_change"]:
                    continue

                new_decimals = decimals
                new_units = units
                patience = 5
                for p in range(patience):
                    self.set_random_seed(initial_offset+p)
                    self.perturber.set_random_seed(initial_offset+p)
                    good = True
                    fk_column = None
                    value_to_keep = None
                    cols_to_avoid = []
                    list_of_data = []
                    list_of_table_hct = []
                    list_of_table_rel = []

                    for i in range(num_tables):
                        new_value_col = deepcopy(value_col)

                        # extracting schema view
                        table_attributes_view, table_attributes_long_view, table_types_view, range_view = self.get_table_view_fk(
                            table_attributes,
                            table_attributes_long,
                            table_types,
                            ranges,
                            new_value_col,
                            fk_column,
                            cols_to_avoid,
                            is_final = True if i == num_tables-1 else False,
                        )

                        # extracting value to put in the middle of the table
                        if i != num_tables-1:
                            new_value_col = self.rnd.choice([attr for h, attr in enumerate(table_attributes_view) if attr != fk_column and table_types_view[h] not in ["float", "int"]])

                        # latent relational source generation
                        try:
                            view_constraints = {"inter_row_constraints": [], "intra_row_constraints": []}
                            for constraint_type in constraints:
                                for rule in constraints[constraint_type]:
                                    ok = True
                                    if constraint_type == "inter_row_constraints":
                                        conditions = [m.group("cond") for m in re.compile(self.constrainer.ATOM_EXTRACT,
                                                                                          re.VERBOSE | re.DOTALL).finditer(rule)]
                                    else:
                                        m = self.constrainer.PATTERN_intra.match(rule)
                                        if not m:
                                            raise ValueError(f"Invalid intra rule: {rule!r}")
                                        conditions = [m.group("condition").strip()]
                                        expr = [m.group("expr").strip()]
                                        conditions.extend(expr)

                                    for cond in conditions:
                                        cond = cond.strip()
                                        if cond[0] in ["(", ")"]:
                                            cond = cond[1:]
                                        if cond[-1] in ["(", ")"]:
                                            cond = cond[:-1]
                                        columns_from_rule = extract_columns(parse_expr(cond))
                                        for col in columns_from_rule:
                                            if col not in table_attributes_view:
                                                ok = False
                                                break
                                        if not ok:
                                            break

                                    if ok:
                                        view_constraints[constraint_type].append(rule)

                            table = self.fill_dense_relational_table(
                                table_attributes_view,
                                table_types_view,
                                range_view,
                                units,
                                decimals,
                                new_value_col, #value_col,
                                semantic_constraints=view_constraints,
                                col_to_keep=fk_column,
                                value_to_keep=value_to_keep,
                            )
                            table = table[table_attributes_view].drop_duplicates(keep="first")
                            new_table = table.copy()

                            if fk_column is not None and i != num_tables-1:
                                mask = new_table[fk_column].eq(value_to_keep)
                                idx = new_table.index[mask]
                                to_drop = idx[1:]
                                new_table = new_table.drop(to_drop)
                                table = table.drop(to_drop)

                        except Exception as e:
                            errors.append("error in filling table")
                            good = False
                            break

                        # run sql loading and sql generation/execution
                        if i != num_tables-1:
                            method_single_table = "extractive"
                        else:
                            method_single_table = method_name

                        data, full_mask = self.generate_label(
                            new_table,
                            table_name,
                            new_value_col,
                            method=method_single_table,
                            col_to_keep=fk_column,
                            value_to_keep=value_to_keep,
                            impose_target_for_extractive=True if method_single_table == "extractive" else False
                        )

                        if data is None:
                            good = False
                            errors.append(f"error in data generation")
                            break

                        fk_column = data["target"] # column used later to connect to the next table
                        value_to_keep = data["label"]
                        cols_to_avoid.extend([col for col in new_table.columns if col != fk_column])

                        new_full_mask = full_mask
                        new_table_types = deepcopy(table_types)

                        if self.rnd.randint(0,1) == 1: # apply null perturbation with 50% chance
                            attrs = {
                                "table": new_table,
                                "constraints": list(zip(data["columns"], data["constraint"])),
                                "value_col": new_value_col,
                            }
                            new_table = self.perturber.null_perturbation(**attrs)

                        # substitute table names with more verbose ones, so to resemble more web-tables
                        try:
                            new_table = self.substitute_column_names(new_table, table_attributes_long_view)
                            new_value_col = table_attributes_long[table_attributes.index(new_value_col)]
                        except Exception as e:
                            good = False
                            errors.append("column name substitution error")
                            break

                        pos_col1, pos_col2 = None, None
                        # apply pre perturbations
                        for perturbation in self.perturber.pre_hct_perturbations:
                            if self.rnd.randint(0,1) == 1 and i != 0:
                                attrs = {
                                    "table": new_table,
                                    "value_col": new_value_col,
                                    "columns": data["columns"],
                                    "constraints": data["constraint"],
                                    "fk": True
                                }

                                new_table, new_value_col, pos_col1, pos_col2 = perturbation(**attrs)

                        # add unit of measurement and apply pivot
                        try:
                            if i != num_tables - 1:
                                unit_in_cell = None
                            else:
                                unit_in_cell = self.rnd.randint(0, 1)
                                new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, new_units,
                                                                                                     unit_in_cell=unit_in_cell)

                            table_before_pivot = new_table.copy()
                            table_hct, rows_chosen, cols_chosen, option, kept_rows, kept_cols, kept_full = self.perturber.multiheader_perturbation(new_table, new_value_col,
                                                                                aggr=aggr, fk=True,
                                                                                unit_in_cell=unit_in_cell, full_mask=new_full_mask)
                        except:
                            good = False
                            errors.append("error pivoting")
                            break

                        if table_hct is None:
                            good = False
                            errors.append("table generation error (pivoting)")
                            continue

                        if pos_col1 is not None:
                            last_col = kept_full[:, -1].copy()
                            for k, pos in enumerate(sorted([pos_col1, pos_col2])):
                                kept_full = np.insert(kept_full, pos + k, last_col, axis=1)
                            kept_full = np.delete(kept_full, -1, axis=1)

                        # we reduce the SQL query WHERE constraints, so to make question generation more concise
                        data["query"] = self.reduce_sql_where(data["query"], table, kept_full)

                        table_hct = self.perturber.restore_needed_cells_after_value_merge(
                            table_before_pivot=table_before_pivot,
                            pivot_table=table_hct,
                            value_col=new_value_col,
                            full_mask=full_mask,
                            rows_chosen=rows_chosen,
                            cols_chosen=cols_chosen,
                        )

                        # applying post pivot perturbations
                        for perturbation in self.perturber.post_hct_perturbations:
                            if perturbation == self.perturber.insert_blank_columns and option == 0: # may cancel unit of measurement
                                continue
                            if self.rnd.randint(0,1) == 1:
                                attrs = {
                                    "table": table_hct
                                }

                                table_hct = perturbation(**attrs)

                        data["decimals"] = new_decimals[new_units[0]]
                        data["units"] = new_units[0]
                        list_of_table_rel.append(table.copy())
                        list_of_table_hct.append(table_hct.copy())
                        list_of_data.append(deepcopy(data))

                    if good:
                        break

                if not good:
                    continue

                if method_name not in datasets[str(num_tables)]:
                    datasets[str(num_tables)][method_name] = {}

                # generating final label
                data = self.generate_label_multitable_fk(list_of_data, method=method_name)
                if data is None:
                    errors.append("error in generating multitable data")
                    continue

                # generating nl question
                data["nl_question"], tables_final = self.generate_question_multitable_fk(
                    list_of_table_hct,
                    data,
                    [d["decimals"] for d in list_of_data],
                    method_name,
                )

                if data["nl_question"] is None:
                    errors.append("error in generating nl question")
                    continue

                # checking nl question
                data["nl_question"] = self.check_nlquestion_validity_fk(
                    data["nl_question"],
                    data,
                    tables_final,
                    data["label"],
                    multi=True,
                    unit=list_of_data[0]["units"]
                )
                if data["nl_question"] is None:
                    errors.append("error in checking nl question")
                    continue

                data["decimals"] = [d["decimals"] for d in list_of_data]
                random.shuffle(list_of_table_hct) # shuffling the table order to increase complexity
                datasets[str(num_tables)][method_name]["sequential"] = (list_of_table_hct, data, constraints)

        return datasets, decimals[units[0]], "\n".join(errors) if len(errors) > 0 else None

    def run_generation(self, num_tables: int = -1, num_samples: int = 1, domain: str | None = None, sequential: bool = False, method: str | None = None, col_cardinality: int = 8, num_columns: int = 6):
        if num_tables == 1:
            datasets, datasets_df, error_logs = {}, {}, []
            perturbations_to_apply = [self.perturber.null_perturbation] + \
                                     self.perturber.pre_hct_perturbations + \
                                     self.perturber.post_hct_perturbations

            for method in SQL_TEMPLATES:
                datasets[method] = {}
                datasets_df[method] = {}
                for perturbation in perturbations_to_apply:
                    try:
                        perturbation_name = perturbation.__name__
                    except:
                        perturbation_name = perturbation

                    datasets[method][perturbation_name] = []

            for _ in tqdm(range(num_samples), desc="Generating samples..."):
                result, decimals, error_log = self.run_one_table_ablations(domain=domain, method=method, col_cardinality=col_cardinality, num_columns=num_columns)
                if result is None:
                    print(f"Error in sample generation: {error_log}")
                    error_logs.append(error_log)
                    continue
                else:
                    print(result)

                for method in SQL_TEMPLATES:
                    for perturbation in perturbations_to_apply:
                        try:
                            perturbation_name = perturbation.__name__
                        except:
                            perturbation_name = perturbation

                        try:
                            if method in result and perturbation_name in result[method]:
                                with pd.option_context("display.float_format", lambda x: f"{x:.{decimals}f}"):
                                    datasets[method][perturbation_name].append([
                                        result[method][perturbation_name][1]["nl_question"],
                                        result[method][perturbation_name][1]["query"],
                                        method,
                                        result[method][perturbation_name][0].to_html(index=True),
                                        result[method][perturbation_name][2],
                                        result[method][perturbation_name][1]["label"],
                                    ])
                        except:
                            continue

            for k1 in datasets:
                for k2 in datasets[k1]:
                    datasets_df[k1][k2] = pd.DataFrame(datasets[k1][k2], columns=["Question", "SQL Query", "Method", "Table", "Constraints", "Label"])

            return datasets_df, error_logs
        else:
            datasets, datasets_df, error_logs = {}, {}, {}
            if not sequential:
                perturbation_names = ["unit_converted", "not_unit_converted"]
                if num_tables != -1:
                    ranges = [num_tables]
                else:
                    ranges = [2,3,5,10,20]
            else:
                perturbation_names = ["sequential"]
                if num_tables != -1:
                    ranges = [num_tables]
                else:
                    ranges = [2,3,5]

            # setting up data containers
            for num_tables_r in ranges:
                datasets[str(num_tables_r)] = {}
                datasets_df[str(num_tables_r)] = {}
                error_logs[str(num_tables_r)] = []

                for method_tmp in SQL_TEMPLATES:
                    if not sequential and method_tmp in ["extractive", "comparative", "percentage_change"]:
                        continue
                    datasets[str(num_tables_r)][method_tmp] = {}
                    datasets_df[str(num_tables_r)][method_tmp] = {}

                    for perturbation_name in perturbation_names:
                        datasets[str(num_tables_r)][method_tmp][perturbation_name] = []

            # generating samples...
            for _ in tqdm(range(num_samples), desc="Generating samples..."):
                try:
                    if num_tables != -1:
                        if sequential:
                            result, decimals, error_log = self.run_multi_table_ablations_fk(domain=domain, num_tables=num_tables, method=method, col_cardinality=col_cardinality, num_columns=num_columns)
                        else:
                            result, decimals, error_log = self.run_multi_table_ablations(domain=domain, num_tables=num_tables, method=method, num_columns=num_columns)
                    else:
                        if sequential:
                            result, decimals, error_log = self.run_multi_table_ablations_fk(domain=domain, method=method)
                        else:
                            result, decimals, error_log = self.run_multi_table_ablations(domain=domain, method=method)

                except:
                    continue

                if result is None:
                    print(f"Error in sample generation: {error_log}")
                    error_logs[str(num_tables)].append(error_log)
                    continue
                else:
                    if error_log is not None:
                        print(f"Error in sample generation: {error_log}")
                        error_logs[str(num_tables)].append(error_log)

                if method is not None:
                    method_list = [method]
                else:
                    method_list = SQL_TEMPLATES

                for num_tables in ranges:
                    for method in method_list:
                        if not sequential and method in ["extractive", "comparative", "percentage_change"]: # skipping single-table queries
                            continue

                        for perturbation_name in perturbation_names:
                            try:
                                if method in result[str(num_tables)] and perturbation_name in result[str(num_tables)][method]:
                                    text = ""
                                    for i, table in enumerate(result[str(num_tables)][method][perturbation_name][0]):
                                        with pd.option_context("display.float_format", lambda x: f"{x:.{result[str(num_tables)][method][perturbation_name][1]['decimals'][i]}f}"):
                                            text += f"Table {i}\n\n{table.to_html(index=True)}\n\n"
                                    text = text.strip()

                                    datasets[str(num_tables)][method][perturbation_name].append([
                                        result[str(num_tables)][method][perturbation_name][1]["nl_question"],
                                        result[str(num_tables)][method][perturbation_name][1]["query"],
                                        method,
                                        text,
                                        result[str(num_tables)][method][perturbation_name][2],  # constraints
                                        result[str(num_tables)][method][perturbation_name][1]["label"],
                                    ])
                            except:
                                continue
                print("Current lengths:")
                for nt in datasets:
                    for m in datasets[nt]:
                        if "unit_converted" in datasets[nt][m]:
                            print(f"\t {nt} tables: {len(datasets[nt][m]["unit_converted"])}")
                        else:
                            print(f"\t {nt} tables: {len(datasets[nt][m]["sequential"])}")
                """if 'unit_converted' in datasets['2']['average'] or 'not_unit_converted' in datasets['2']['average']:
                    print(f"\t 2 tables: {len(datasets['2']['average']['unit_converted'])}")
                    print(f"\t 3 tables: {len(datasets['3']['average']['unit_converted'])}")
                    print(f"\t 5 tables: {len(datasets['5']['average']['unit_converted'])}")
                    print(f"\t 10 tables: {len(datasets['10']['average']['unit_converted'])}")
                    print(f"\t 20 tables: {len(datasets['20']['average']['unit_converted'])}")
                else:
                    print(f"\t 2 tables: {len(datasets['2']['average']['sequential'])}")
                    print(f"\t 3 tables: {len(datasets['3']['average']['sequential'])}")
                    print(f"\t 5 tables: {len(datasets['5']['average']['sequential'])}")"""

            for num_tables in datasets:
                for k1 in datasets[str(num_tables)]:
                    for k2 in datasets[str(num_tables)][k1]:
                        datasets_df[str(num_tables)][k1][k2] = pd.DataFrame(datasets[str(num_tables)][k1][k2],
                                                           columns=["Question", "SQL Query", "Method", "Table",
                                                                    "Constraints", "Label"])

            return datasets_df, error_logs

    def run_real_comparison(self, num_tables: int = 1, question_type: str = "parallel"):
        from table_extraction_utils import find_densest_pivot, extract_tables_from_sqlite_directories
        import pickle

        tables_dataset = pd.read_csv("pivot_task/exports/qualifying_dense_pivots_verified_moredbs.csv")
        tables_dataset = pd.concat(
            [
                tables_dataset.loc[
                    ((tables_dataset["pivot_rows"] >= 10) | (tables_dataset["pivot_columns"] >= 10)) &
                    ((tables_dataset["num_row_attributes"] > 2) | (tables_dataset["num_column_attributes"] > 2)) &
                    ((tables_dataset["pivot_rows"] > 3) & (tables_dataset["pivot_columns"] > 3))
                    ]
                .groupby(["dataset", "relation"], group_keys=False)
                .apply(lambda g: g.sample(n=min(10, len(g)), random_state=42)),

                tables_dataset.loc[
                    ((tables_dataset["pivot_rows"] >= 10) | (tables_dataset["pivot_columns"] >= 10)) &
                    ((tables_dataset["num_row_attributes"] > 1) | (tables_dataset["num_column_attributes"] > 1)) &
                    ((tables_dataset["pivot_rows"] > 2) & (tables_dataset["pivot_columns"] > 2))
                    ]
                .groupby(["dataset", "relation"], group_keys=False)
                .apply(lambda g: g.sample(n=min(5, len(g)), random_state=42)),
            ],
            ignore_index=True,
        ).reset_index(drop=True)

        print(f"*** NUM TABLES: {len(tables_dataset)} ***")

        categorical_cols_per_table = tables_dataset["row_attributes"].apply(lambda x: [el.strip() for el in x.split(";")])
        categorical_cols_per_table += tables_dataset["column_attributes"].apply(lambda x: [el.strip() for el in x.split(";")])
        float_cols_per_table = tables_dataset["value_attribute"]
        cols_split = tables_dataset["row_attributes"].apply(lambda x: [[el.strip() for el in x.split(";")]])
        cols_split += tables_dataset["column_attributes"].apply(lambda x: [[el.strip() for el in x.split(";")]])
        tables = [pd.read_csv(path) for path in tables_dataset["table_csv_path"]]

        path = "datasets/real/"
        os.makedirs(path, exist_ok=True)

        datasets = {}
        ccols, fcols = categorical_cols_per_table, float_cols_per_table

        tables_relational, tables_non_relational = [], []

        for num, (table, ccol, fcol, c_split) in tqdm(enumerate(zip(tables, ccols, fcols, cols_split))):
            if num == 9:
                continue
            # tables may have whitespaces and brackets, that could potentially disrupt SQL templates. For safety, we remove them
            cleaner = lambda x: x.replace(" ", "").replace("(", "").replace(")", "")
            table.columns = table.columns.map(cleaner)
            ccol = [cleaner(c) for c in ccol]
            fcol = cleaner(fcol)

            cols = c_split[0] + c_split[1]
            decimals = len(str(table[fcol].iloc[0]).split(".")[-1])

            table_relational = table.drop_duplicates(subset=cols, keep="first")

            tables_relational.append(table_relational)
            new_table = table_relational.copy()
            new_value_col = fcol

            tables_unit_converted, tables_not_unit_converted, tables_relational_final = [], [], []
            data_unit_converted, data_not_unit_converted, data_relational = [], [], []
            units, decimals = ["unit"], len(str(table[fcol].iloc[0]).split(".")[-1])

            if num_tables == 1:
                first = True
                for j, method in enumerate(SQL_TEMPLATES):
                    if method not in ["average", "sum", "superlative"]:
                        continue

                    new_table = table_relational[cols + [fcol]].copy()
                    new_table = new_table.drop_duplicates(subset=cols, keep="first")
                    new_value_col = fcol
                    new_table[new_value_col] = new_table[new_value_col].apply(lambda x: round(x, decimals))
                    table_relational_to_load = new_table.copy()

                    if method not in datasets:
                        datasets[method] = {}

                    try:
                        data, full_mask = self.generate_label(
                            new_table,
                            "sample_table",
                            new_value_col,
                            method=method,
                        )
                    except Exception as e:
                        continue

                    if data is None:
                        continue

                    # enforce at most 100 cells, while preserving all rows needed for the answer
                    if table_relational_to_load.shape[0] * table_relational_to_load.shape[1] > 100:
                        required_mask = full_mask.loc[table_relational_to_load.index]
                        max_rows = max(1, 100 // table_relational_to_load.shape[1])

                        if required_mask.sum() <= max_rows:
                            kept_mask = required_mask.copy()
                            extra_needed = max_rows - required_mask.sum()

                            if extra_needed > 0:
                                extra_indices = table_relational_to_load.index[~required_mask][:extra_needed]
                                kept_mask.loc[extra_indices] = True

                            table_relational_to_load = table_relational_to_load.loc[kept_mask]
                        else:
                            # keep all required rows in their original order
                            table_relational_to_load = table_relational_to_load.loc[required_mask]

                    if self.rnd.randint(1, 1) == 1:  # apply perturbation with 50% chance
                        attrs = {
                            "table": new_table,
                            "constraints": list(zip(data["columns"], data["constraint"])),
                            "value_col": new_value_col,
                        }
                        new_table = self.perturber.null_perturbation(**attrs)


                    for perturbation in self.perturber.pre_hct_perturbations:
                        if self.rnd.randint(1, 1) == 1:
                            attrs = {
                                "table": new_table,
                                "value_col": new_value_col
                            }

                            new_table, new_value_col = perturbation(**attrs)


                    try:
                        unit_in_cell = self.rnd.randint(0, 1)
                        new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col,
                                                                                             ["unit"],
                                                                                             unit_in_cell=unit_in_cell)
                        table_before_pivot = new_table.copy()
                        table_hct, rows_chosen, cols_chosen, _ = self.perturber.multiheader_perturbation(new_table,
                                                                                                         new_value_col,
                                                                                                         #"",
                                                                                                         #[],
                                                                                                         aggr="first",
                                                                                                         unit_in_cell=unit_in_cell,
                                                                                                         full_mask=full_mask)
                    except:
                        continue

                    if table_hct is None:
                        continue


                    table_hct = self.perturber.restore_needed_cells_after_value_merge(
                        table_before_pivot=table_before_pivot,
                        pivot_table=table_hct,
                        value_col=new_value_col,
                        full_mask=full_mask,
                        rows_chosen=rows_chosen,
                        cols_chosen=cols_chosen,
                    )

                    for perturbation in self.perturber.post_hct_perturbations:
                        if self.rnd.randint(1, 1) == 1:
                            attrs = {
                                "table": table_hct
                            }

                            table_hct = perturbation(**attrs)


                    data["nl_question"] = self.generate_question(table_hct, data, decimals, method=method)
                    if data["nl_question"] is None:
                        continue


                    with pd.option_context("display.float_format", lambda x: f"{x:.{decimals}f}"):
                        data["nl_question"] = self.check_nlquestion_validity(data["nl_question"], data["query"],
                                                                             table_hct.to_html(index=False), data["label"])
                    if data["nl_question"] is None:
                        continue

                    if "non_relational" not in datasets[method]:
                        datasets[method]["non_relational"] = []
                    if "relational" not in datasets[method]:
                        datasets[method]["relational"] = []

                    data["decimals"] = decimals
                    datasets[method]["non_relational"].append((table_hct, data))
                    datasets[method]["relational"].append((table_relational_to_load, data))

                    if first:
                        print(f"Single table sample number {num} added to dataset")
                        first = False
            elif question_type == "parallel":
                columns, constraint_cols, column_to_vary, old_col_values, best_val = None, None, None, [], None
                old_constraint_cols = None
                good = True
                table = table.drop_duplicates(subset=cols)

                for i in range(num_tables):
                    table_rel = table.copy()
                    table_rel = table_rel[cols+[fcol]]
                    table_rel = table_rel.drop_duplicates(subset=cols)
                    table_rel[cols] = table_rel[cols].fillna("None")
                    table_rel = table_rel.dropna(subset=[fcol])
                    new_value_col = fcol
                    table_rel[new_value_col] = table_rel[new_value_col].apply(lambda x: round(x, decimals))
                    table_relational_to_load = table_rel.copy()

                    if i != 0:
                        try:
                            best_val = self.choose_constraint_on_new_table(
                                columns,
                                column_to_vary,
                                old_col_values,
                                table_rel,
                            )

                            if old_constraint_cols is not None:
                                constraint_cols = deepcopy(old_constraint_cols)
                                old_constraint_cols = None

                            if isinstance(constraint_cols, tuple):
                                constraint_cols = list(constraint_cols)

                            constraint_cols[column_to_vary] = best_val
                            if len(table_rel[(table_rel[columns] == constraint_cols).all(axis=1)]) == 0:
                                found = False
                                for col_to_change in [i for i in range(len(columns)) if i != column_to_vary]:
                                    #col_to_change = random.choice([i for i in range(len(columns)) if i != column_to_vary])
                                    columns_without_col_to_change = [c for j,c in enumerate(columns) if j != col_to_change]
                                    constraint_cols_without_col_to_change = [c for j,c in enumerate(constraint_cols) if j != col_to_change]
                                    solutions = table_rel[
                                        (table_rel[columns_without_col_to_change] == constraint_cols_without_col_to_change).all(axis=1)
                                    ].reset_index()
                                    old_constraint_cols = deepcopy(constraint_cols)
                                    if len(solutions) > 0:
                                        constraint_cols[col_to_change] = solutions.loc[0,columns[col_to_change]]
                                        found = True
                                        break

                                if not found:
                                    good = False
                                    break

                            # rows where the selected columns match the given values
                            mask = (table_rel[columns] == constraint_cols).all(axis=1)

                            # keep all non-matching rows, and deduplicate matching rows
                            table_relational_to_load = pd.concat([
                                table_rel[~mask],
                                table_rel[mask].drop_duplicates(subset=columns, keep="first")
                            ], ignore_index=True)

                            table_rel = pd.concat([
                                table_rel[~mask],
                                table_rel[mask].drop_duplicates(subset=columns, keep="first")
                            ], ignore_index=True)
                        except:
                            good = False
                            break

                    try:
                        data, full_mask = self.generate_label(
                            table_rel,
                            "sample_table",
                            new_value_col,
                            method="extractive", # we force the method to be extractive, as the true method will be applied across tables later
                            columns=columns,
                            constraint=constraint_cols,
                            column_to_vary=column_to_vary,
                            old_col_values=old_col_values,
                            best_val=best_val,
                            impose_target_for_extractive=True,
                            #all_cols=True
                        )
                    except:
                        good = False
                        break

                    if data is None:
                        good = False
                        break

                    if columns is None and constraint_cols is None:
                        columns = data["columns"]
                        constraint_cols = data["constraint"]

                        # I make sure that inter row constraints are kept even when dealing with multiple tables
                        # for that, in different tables, I vary an attribute that is not part of the inter-row constraints (if any)
                        columns_to_consider = [col for col in columns]
                        # the column to vary must not be the value_col
                        columns_to_consider = [col for col in columns_to_consider if col != new_value_col]
                        # the column to vary must have at least num_tables different values, to ensure that we can vary it across tables
                        columns_to_consider = [col for col in columns_to_consider if table_rel[col].nunique() >= num_tables]
                        if len(columns_to_consider) == 0:
                            good = False
                            break

                        max_val, best_col = -1, None
                        for col in columns_to_consider:
                            pos = columns.index(col)
                            num_samples = len(table_rel[table_rel[columns[pos]] == data["constraint"][pos]])
                            if num_samples > max_val:
                                max_val = num_samples
                                best_col = col

                        column_to_vary = columns.index(best_col)

                    old_col_values.append(data["constraint"][column_to_vary])

                    table_rel = table_rel[table_rel[columns[column_to_vary]] == data["constraint"][column_to_vary]]
                    selection_rule = table_relational_to_load[columns[column_to_vary]] == data["constraint"][column_to_vary]
                    table_relational_to_load = table_relational_to_load[selection_rule]

                    full_mask = full_mask[selection_rule]

                    if table_relational_to_load.shape[0] * table_relational_to_load.shape[1] > 100:
                        required_mask = full_mask.loc[table_relational_to_load.index]
                        max_rows = max(1, 100 // table_relational_to_load.shape[1])

                        if required_mask.sum() <= max_rows:
                            kept_mask = required_mask.copy()
                            extra_needed = max_rows - required_mask.sum()

                            if extra_needed > 0:
                                extra_indices = table_relational_to_load.index[~required_mask][:extra_needed]
                                kept_mask.loc[extra_indices] = True

                            table_relational_to_load = table_relational_to_load.loc[kept_mask]
                        else:
                            # keep all required rows in their original order
                            table_relational_to_load = table_relational_to_load.loc[required_mask]

                    # adding a sample with all perturbations randomly applied with lower strength
                    for apply_unit_conversions in [True, False]:
                        self.set_random_seed(num+i)
                        self.perturber.set_random_seed(num+i)
                        new_table = table_rel.copy()
                        new_full_mask = full_mask
                        new_value_col = fcol

                        try:
                            if i != 0 and apply_unit_conversions:
                                needed_decimals = -1
                                for j in range(5):
                                    surrogate_unit, target_unit = get_random_unit(units[0], "finance") #using financial domain to use millions, thousands etc. notation
                                    highest_needed_decimals = -1
                                    for k, row in new_table.iterrows():
                                        try:
                                            value_converted, needed_decimals = get_value(
                                                surrogate_unit,
                                                target_unit,
                                                "finance",
                                                round(row[new_value_col], 10), # avoid strange infinite decimals
                                                self.perturber.nan_fill_str
                                            )
                                        except:
                                            good = False
                                            break

                                        if needed_decimals is not None:
                                            if needed_decimals == -1:
                                                break
                                                # return None, None, "couldn't find rounding with the specified number of decimal values"

                                            # new_table.at[k, new_value_col] = round(value_converted, needed_decimals)
                                            new_table.at[k, new_value_col] = value_converted
                                            if needed_decimals > highest_needed_decimals:
                                                highest_needed_decimals = needed_decimals

                                    if needed_decimals != -1:
                                        #good = False
                                        break

                                if needed_decimals == -1:
                                    good = False
                                    break

                                # data["label"] = get_value(surrogate_unit, target_unit, domain, data["label"], self.perturber.nan_fill_str)
                                if surrogate_unit != units[0]:
                                    target_unit = f"{target_unit} {units[0].replace('[', '').replace(']', '')}"

                                new_units = [target_unit]
                                new_decimals = max(decimals, highest_needed_decimals) #{target_unit: max(decimals, highest_needed_decimals)}
                            else:
                                new_units = ["units"]
                                new_decimals = decimals
                        except:
                            good = False
                            break

                        attrs = {
                            "table": new_table,
                            "constraints": list(zip(data["columns"], data["constraint"])), #{k: v for k, v in zip(data["columns"], data["constraint"])},
                            "value_col": new_value_col,
                        }
                        new_table = self.perturber.null_perturbation(**attrs)

                        if i == 0:
                            rows_chosen = deepcopy(c_split[0])
                            cols_chosen = deepcopy(c_split[1])
                        elif i == 1:
                            rows_chosen = deepcopy(c_split[1])
                            cols_chosen = deepcopy(c_split[0])
                        else:
                            rows_chosen = self.rnd.sample(c_split[0], k=len(c_split[0]))
                            cols_chosen = self.rnd.sample(c_split[1], k=len(c_split[1]))

                        for perturbation in self.perturber.pre_hct_perturbations:
                            if i != 0:
                                attrs = {
                                    "table": new_table,
                                    "value_col": new_value_col,
                                    "return_cols": True,
                                    "rows_chosen": rows_chosen,
                                    "cols_chosen": cols_chosen,
                                }

                                new_table, new_value_col, col1, col2, is_value_col = perturbation(**attrs)
                            else:
                                col1, col2 = None, None
                                is_value_col = True

                        try:
                            unit_in_cell = self.rnd.randint(0, 1)
                            new_table, new_value_col = self.perturber.insert_unit_of_measurement(new_table, new_value_col, new_units,
                                                                                                 unit_in_cell=unit_in_cell)
                            table_before_pivot = new_table.copy()

                            if col1 and col1 in rows_chosen:
                                rows_chosen.remove(col1)
                            if col1 and col1 in cols_chosen:
                                cols_chosen.remove(col1)
                            if col2 and col2 in rows_chosen:
                                rows_chosen.remove(col2)
                            if col2 and col2 in cols_chosen:
                                cols_chosen.remove(col2)

                            if not is_value_col:
                                rows_chosen.append(f"{col1} ({col2})")

                            table_hct, rows_chosen, cols_chosen, option = self.perturber.multiheader_perturbation(new_table, new_value_col,
                                                                                #"", [],
                                                                                aggr="first",
                                                                                unit_in_cell=unit_in_cell, full_mask=new_full_mask,
                                                                                rows_chosen=rows_chosen, cols_chosen=cols_chosen)

                        except:
                            good = False
                            break

                        if table_hct is None:
                            good = False
                            break

                        table_hct = self.perturber.restore_needed_cells_after_value_merge(
                            table_before_pivot=table_before_pivot,
                            pivot_table=table_hct,
                            value_col=new_value_col,
                            full_mask=full_mask,
                            rows_chosen=rows_chosen,
                            cols_chosen=cols_chosen,
                        )

                        for perturbation in self.perturber.post_hct_perturbations:
                            if perturbation == self.perturber.insert_blank_columns and option == 0: # may cancel unit of measurement
                                continue

                            attrs = {
                                "table": table_hct
                            }

                            table_hct = perturbation(**attrs)

                        data["decimals"] = new_decimals
                        data["units"] = new_units[0]
                        if apply_unit_conversions:
                            tables_unit_converted.append(table_hct.copy())
                            data_unit_converted.append(deepcopy(data))
                        else:
                            tables_not_unit_converted.append(table_hct.copy())
                            data_not_unit_converted.append(deepcopy(data))

                    tables_relational_final.append(table_relational_to_load.copy())
                    data_relational.append(deepcopy(data_unit_converted[-1]))

                if not good:
                    continue

                first = True
                for num_question, (perturbation_name, list_of_table_hct, list_of_data) in enumerate(zip(
                        ["unit_converted", "not_unit_converted", "relational_multi_table"],
                        [tables_unit_converted, tables_not_unit_converted, tables_relational_final],
                        [data_unit_converted, data_not_unit_converted, data_relational]
                )):

                    for method in SQL_TEMPLATES:
                        if method not in ["average", "sum", "superlative"]:
                            continue
                        if method not in datasets:
                            datasets[method] = {}

                        data = self.generate_label_multitable(list_of_data, method=method)
                        if data is None:
                            continue

                        data["nl_question"], tables_final = self.generate_question_multitable(
                            list_of_table_hct,
                            data,
                            [d["decimals"] for d in list_of_data],
                            method,
                            unit=list_of_data[0]["units"]
                        )
                        if data["nl_question"] is None:
                            continue

                        data["nl_question"] = self.check_nlquestion_validity(
                            data["nl_question"],
                            data["query"],
                            tables_final,
                            data["label"],
                            multi=True,
                            unit=list_of_data[0]["units"]
                        )
                        if data["nl_question"] is None:
                            continue

                        if perturbation_name not in datasets[method]:
                            datasets[method][perturbation_name] = []

                        data["decimals"] = [d["decimals"] for d in list_of_data]
                        datasets[method][perturbation_name].append((list_of_table_hct, data))

                        if first:
                            print(f"Multi table sample number {num} added to dataset")
                            first = False

        return datasets
