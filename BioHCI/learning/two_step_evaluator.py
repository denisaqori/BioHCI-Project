"""
Created: 1/7/20
© Denisa Qori McDonald 2020 All Rights Reserved
"""
from BioHCI.learning.evaluator import Evaluator


class TwoStepEvaluator(Evaluator):
    def __init__(self, val_data_loader, model_to_eval, criterion, knitted_component, confusion, neural_network_def,
                 parameters, summary_writer):
        self.__knitted_component = knitted_component
        super(TwoStepEvaluator, self).__init__(val_data_loader, model_to_eval, criterion, confusion,
                                               neural_network_def, parameters, summary_writer)

    def __category_from_output(self, output):
        return self.__knitted_component.get_button_id(output)
