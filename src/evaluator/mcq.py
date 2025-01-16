from .base import Evaluator
import numpy as np
import random

class MCQEvaluator(Evaluator):
    def extract_answer(self, response):
        response = response.lower()
        if response.startswith('<1>') or response.startswith('<2>') or response.startswith('<3>'):
            response = response[3:].strip()
        for template in [
            "答案是[CHOICE]",
            "答案是 [CHOICE]",
            "答案是选项[CHOICE]",
            "答案应该是[CHOICE]",
            "答案应该是 [CHOICE]",
            "答案就是选项[CHOICE]",
            "答案是‘[CHOICE]",
            "是[CHOICE]：",
            "答案选[CHOICE]",
            "[CHOICE]是正确",
            "选项[CHOICE]是最合适的",
            'answer is **',
            "the answer is '[CHOICE]'",
            '[CHOICE] is the best answer',
            'the answer is [CHOICE]',
            'the correct answer is [CHOICE]',
            'would select [CHOICE]',
            'would choose [CHOICE]',
            'would select option [CHOICE]',
            'would choose option [CHOICE]',
            'is \"[CHOICE]\"',
            'is \"[CHOICE].',
            'would be [CHOICE]',
            'would be option [CHOICE]',
            'would be ([CHOICE])',
            'would be option ([CHOICE])',
            'is [CHOICE],',
            'is typically [CHOICE],',
            'is typically [CHOICE].',
            "i'd say [CHOICE].",
            "option [CHOICE].",
            "option [CHOICE]:",
            "option [CHOICE],",
            "is [CHOICE]:",
            "is [CHOICE].",
            "is [CHOICE],",
            "is: [CHOICE].",
            "is ([CHOICE])",
            ":\n[CHOICE].",
            ":\n[CHOICE])",
            ":\n[CHOICE],",
            ": \n[CHOICE].",
            ":  \n[CHOICE].",
            ":\n\n[CHOICE].",
            ":\n\n[CHOICE])",
            ":\n\n[CHOICE],",
            ": \n\n[CHOICE].",
            "is option [CHOICE],",
            '([CHOICE]) would be',
            'is ([CHOICE]).',
            "is [CHOICE])",
            "is: [CHOICE])",
            '(option [CHOICE])',
            'answer is ([CHOICE])',
            "select option \"[CHOICE]\"",
            "is: [CHOICE]",
            "is likely '[CHOICE]'",
            "is option '[CHOICE]'",
            "would be '[CHOICE]'",
            "is the **[CHOICE]** ",
            "the answer to the question is '[CHOICE]'",
            "question is **[CHOICE]**",
            "known as '[CHOICE]'",
            "is '[CHOICE])",
            " [CHOICE].",
            " [CHOICE],",
            " [CHOICE]:",
            " [CHOICE])",
            "\"[CHOICE].",
            "\"[CHOICE],",
            "\"[CHOICE]:",
            "([CHOICE])",
            "\"[CHOICE]\"",

        ]:
            for choice in ['a', 'b', 'c', 'd']:
                if template.replace('[CHOICE]', choice) in response:
                    return choice.upper()
        for choice in ['a', 'b', 'c', 'd']:
            if response == choice:
                return choice.upper()
            for punc in ['.', ',', ':', ')']:
                if response.startswith(choice+punc):
                    return choice.upper()

        if 'would be a.' in response:
            return 'A'
        elif 'would be \"a.' in response:
            return 'A'
        elif 'the best option from the given choices would be a scorpion (a)' in response:
            return 'A'
        else:
            print({response})
            print('====')
            return None


    def evaluate(self, data):
        ground_truth = [item['reference'] for item in data]
        preds = [self.extract_answer(item['response']) for item in data]
        cnt = 0
        for idx in range(len(preds)):
            if preds[idx] == None:
                preds[idx] = random.choice(['A', 'B', 'C', 'D'])
                cnt += 1
        correct_predictions = sum([1 for pred, gt in zip(preds, ground_truth) if pred == gt])
        total_predictions = len(ground_truth)
        accuracy = correct_predictions / total_predictions
        return {
            'acc': accuracy * 100, 'fail': 100 * cnt / len(preds)
        }

