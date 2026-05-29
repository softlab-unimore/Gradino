from models import MTAutoGen
from utils import get_args
import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    args = get_args()
    mtautogen = MTAutoGen(args)
    method, domain, num_tables, num_samples, col_cardinality, num_columns = args['question_type'], args['domain'], args['num_tables'], \
                                                                            args['num_samples'], args['col_cardinality'], args['num_columns']
    if num_tables == 1:
        num_table_str = "one-table"
    elif args["sequential"]:
        num_table_str = "multi-table-sequential"
    else:
        num_table_str = "multi-table"

    if domain is None:
        domain = "general_domain"

    path = os.path.join("datasets", num_table_str)
    os.makedirs(path, exist_ok=True)

    if num_tables == -1:
        print(f"Running ablations: generating {num_samples} {'sequential' if args['sequential'] else 'parallel'} instances on {domain} domain")
        samples, error_logs = mtautogen.run_generation(num_tables=-1, num_samples=num_samples, domain=domain, sequential=args["sequential"])
    else:
        print(f"Generating tables: generating {num_samples} {'sequential' if args['sequential'] else 'parallel'} instances on {domain} domain")
        samples, error_logs = mtautogen.run_generation(num_tables=num_tables, method=method, num_samples=num_samples, domain=domain, col_cardinality=col_cardinality, num_columns=num_columns, sequential=args["sequential"])

    print(f"Errors: {error_logs}")
    for num_tables in samples:
        for k1 in samples[str(num_tables)]:
            for k2 in samples[str(num_tables)][k1]:
                path_k1_k2 = os.path.join(path, str(num_tables), domain, k1, k2, "prova", "data.csv")
                os.makedirs(os.path.dirname(path_k1_k2), exist_ok=True)
                samples[str(num_tables)][k1][k2].to_csv(path_k1_k2, index=False)
                print("Data saved to ", path_k1_k2)
