import os
import re
from argparse import ArgumentParser

import pandas as pd

def preprocessing_range(row):
    row['response'] = re.sub(r'(?<=\d),(?=\d)', '', row['response'])
    row['value'] = re.sub(r'(\d+(\.\d+)?)[-–—](\d+(\.\d+)?)', r'\1 - \3', row['value'])
    row['response'] = re.sub(r'(\d+(\.\d+)?)[-–—](\d+(\.\d+)?)', r'\1 - \3', row['response'])
    row['value'] = re.sub(r'[–—]', '-', row['value'])
    row['response'] = re.sub(r'[–—]', '-', row['response'])
    return row

def preprocessing_range_tablellama(row, dataset):
    row['response'] = re.sub(r'[<>]', '', row['response'])

    if dataset == 'extra':
        row['response'] = re.sub(r'[,]', ' -', row['response'])

    row['response'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.3f}", row['response'])
    row['value'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.3f}", row['value'])
    return row

def preprocessing_range_finma(row, dataset):
    if dataset == 'extra':
        row['response'] = row['response'].replace('and', '-')
    else: # rel dataset
        row['response'] = row['response'].replace(' and ', ', ')
    return row

def preprocessing_range_tattllm(row, dataset):
    if dataset == 'extra':
        row['response'] = row['response'].replace('#', ' - ')
        row['value'] = re.sub(r'[-–—]', '-', row['value'])
    elif dataset == 'rel':
        row['response'] = row['response'].replace('#', ', ')
        row['response'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.3f}", row['response'])
        row['value'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.3f}", row['value'])
    return row

def preprocessing_range_openai(row, dataset):
    row['response'] = re.sub(r'[a-zA-Z]+(\d+)', '', row['response'])
    row['response'] = re.sub(r"[a-zA-Z\u2080-\u2089()/€]", "", row['response']).strip(' ')

    #if dataset == 'extra':
    #    row['response'] = re.sub(r'(\d+\.\d+)\s+(\d+\.\d+)', r"\1 - \2", row['response'])
    #else:
    row['response'] = re.sub(r'(\d+)\s+(\d+)', r'\1, \2', row['response'])
    row['response'] = re.sub(r'\s*,\s*|\s+', ', ', row['response'])

    row['response'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.2f}", row['response'])
    row['value'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.2f}", row['value'])
    return row

def check_number(value, response, percentage=False):
    response = response.replace(',', '')
    try:
        if percentage:
            if (abs(float(value)) == abs(float(response)) or
                    abs(float(value)) == abs(float(float(response) * 100))):
                return True
        return float(value) == float(response)
    except ValueError:
        return False

def calculate_and_save_metrics(results, metrics, model, output_path):
    em = (results.loc[results['correct'] == True].shape[0] / results.shape[0])*100
    metrics.loc[len(metrics)] = {'model': model, 'em': round(em, 1)}
    metrics.to_csv(output_path, index=False)
    print(f'EM: {round(em, 1)}')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--filepath', type=str, default='extra')
    args = parser.parse_args()

    metrics_hier, metrics_not_hier, metrics = (pd.DataFrame(columns=['model', 'em']) for _ in range(3))

    results = pd.read_csv(args.filepath)
    results['value'] = results['Label'].astype(str).str.lower()
    results['response'] = results['Prediction'].astype(str).str.lower()

    for i, row in results.iterrows():
        row['response'] = row['response'].split('###')[0]
        row['response'] = row['response'].strip(' |().\n\r')
        row['response'] = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', row['response'])

        percentage = True if '%' in row['Question'] or 'percentage' in row['Question'] else False

        #if args.dataset == 'extra':
        #    range = True if re.match(r'^\d+(\.\d+)?\s*[-–—]\s*\d+(\.\d+)?$', row['value']) else False
        #else: # rel dataset
        range = True if re.match(r'^\s*\d+(\.\d+)?\s*,\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*\s*$', row['value']) else False

        # Check for range
        if range:
            row = preprocessing_range(row)
            """if model == 'tablellama':
                row = preprocessing_range_tablellama(row, args.dataset)
            if model == 'finma':
                row = preprocessing_range_finma(row, args.dataset)
            if model == 'tatllm__end_to_end' or model == 'tatllm__step_wise':
                row = preprocessing_range_tattllm(row, args.dataset)
            if model == 'openai' or model == 'openai_chainofthought':"""
            row = preprocessing_range_openai(row, args.dataset)

        # Check exact match
        if row['value'] == row['response']:
            results.loc[i, 'correct'] = True

        # Check for numbers with <, =, > symbols in both side
        elif row['value'].startswith('<=') and row['response'].startswith('<='):
            results.loc[i, 'correct'] = check_number(row['value'].strip('%<= '), re.sub(r'[a-zA-Z\u2080-\u2089]', '', row['response']).strip('%<= '), percentage=percentage)
        elif row['value'].startswith('>=') and row['response'].startswith('>='):
            results.loc[i, 'correct'] = check_number(row['value'].strip('%>= '), re.sub(r'[a-zA-Z\u2080-\u2089]', '', row['response']).strip('%>= '), percentage=percentage)
        elif row['value'].startswith('<') and row['response'].startswith('<'):
            results.loc[i, 'correct'] = check_number(row['value'].strip('%< '), re.sub(r'[a-zA-Z\u2080-\u2089]', '', row['response']).strip('%< '), percentage=percentage)
        elif row['value'].startswith('=') and row['response'].startswith('='):
            results.loc[i, 'correct'] = check_number(row['value'].strip('%= '), re.sub(r'[a-zA-Z\u2080-\u2089]', '', row['response']).strip('%= '), percentage=percentage)
        elif row['value'].startswith('>') and row['response'].startswith('>'):
            results.loc[i, 'correct'] = check_number(row['value'].strip('%> '), re.sub(r'[a-zA-Z\u2080-\u2089]', '', row['response']).strip('%> '), percentage=percentage)

        # Check for response with multiple words
        elif ((any(r.isalpha() for r in row['response'])) or (any(c in row['response'] for c in ['(', ')']))) and args.dataset == 'extra':
            row['response'] = re.sub(r"(?<=\d)#(?=[a-zA-Z])", " ", row['response'])
            el = [c.strip('%()') for c in row['response'].split(' ')]
            results.loc[i, 'correct'] = any(
                check_number(row['value'].strip('%~'), e.strip('~'), percentage=percentage) for e in el
            )

        elif (any(r.isalpha() for r in row['response'])) and args.dataset == 'rel':
            row['response'] = row['response'].split(' ')[0]
            results.loc[i, 'correct'] = check_number(row['value'].strip('%~'), row['response'].strip('%~'), percentage=percentage
                                                     )
        # Check for numbers with ., % symbols and without <, =, > symbols
        elif (((any(c in row['value'] or c in row['response'] for c in ['.', '%']) and
              (any(c in row['value'] for c in ['<', '=', '>']) == any(c in row['response'] for c in ['<', '=', '>'])))) or
              any(c in row['value'] or c in row['response'] for c in ['~', ','])):
            results.loc[i, 'correct'] = check_number(row['value'].strip('%<= >~'), row['response'].strip('%<= >~'), percentage=percentage)

        # Otherwise
        else:
            results.loc[i, 'correct'] = False

    results.to_csv(os.path.join('/'.join(args.filepath.split('/')[:-1]), 'preds_with_match.csv'), index=False)
    #results.to_csv(f'./results/{args.type}/{args.dataset}/with_match/{model}.csv', index=False)

    calculate_and_save_metrics(results, metrics, os.path.join('/'.join(args.filepath.split('/')[:-1]), 'metrics.csv'))