# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluate predictions."""
import sys

from utils.evaluate.utils import eval_utils


def NQ_evaluate(references_path, predictions_path, is_regex=False, answer_field="answer"):
    metrics = eval_utils.evaluate_predictions(references_path,
                                              predictions_path,
                                              is_regex, "answer")
    print("Found {} missing predictions.".format(metrics["missing_predictions"]))
    print("Accuracy: {:.2f} ({}/{})".format(metrics["accuracy"] * 100,
                                            metrics["num_correct"],
                                            metrics["num_total"]))
    return metrics["num_correct"], metrics["num_total"], metrics["accuracy"]


if __name__ == "__main__":
    references_path = sys.argv[1]
    if len(sys.argv) == 3:
        predictions_path = sys.argv[2]
    else:
        predictions_path = sys.argv[1]
    NQ_evaluate(references_path, predictions_path)
