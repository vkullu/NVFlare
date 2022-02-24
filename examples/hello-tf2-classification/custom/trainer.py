# Copyright (c) 2021, NVIDIA CORPORATION.
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

import tensorflow as tf
from IPython import embed
import numpy as np
import pandas as pd
import utils


from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal

feature_dim = 0

class SimpleTrainer(Executor):
    def __init__(self, epochs_per_round):
        super().__init__()
        self.epochs_per_round = epochs_per_round
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)

    def setup(self, fl_ctx: FLContext):
        self.X_train, self.X_test, self.y_train, self.y_test = utils.load_cancer_dataset()
        # simulate separate datasets for each client by dividing cancer dataset in half
        client_name = fl_ctx.get_identity_name()
        if client_name == "site-1":
            self.X_train = self.X_train[: len(self.X_train) // 2]
            self.y_train = self.y_train[: len(self.y_train) // 2]
            self.X_test = self.X_test[: len(self.X_test) // 2]
            self.y_test = self.y_test[: len(self.y_test) // 2]
        elif client_name == "site-2":
            self.X_train = self.X_train[len(self.X_train) // 2 :]
            self.y_train = self.y_train[len(self.y_train) // 2 :]
            self.X_test = self.X_test[len(self.X_test) // 2 :]
            self.y_test = self.y_test[len(self.y_test) // 2 :]

        # I am using the simplest model here with tiny data! So don't judge the accuracy :)
        feature_dim = self.X_train.shape[1]
        self.model = utils.get_simple_sequential_model(feature_dim=self.X_train.shape[1])
        self.var_list = [self.model.get_layer(index=index).name for index in range(len(self.model.get_weights()))]

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After finishing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            task_name: dispatched task
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """

        # retrieve model weights download from server's shareable
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        if task_name != "train":
            return make_reply(ReturnCode.TASK_UNKNOWN)

        dxo = from_shareable(shareable)
        model_weights = dxo.data

        # use previous round's client weights to replace excluded layers from server
        prev_weights = {
            self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights()) # these are old weights
        }

        ordered_model_weights = {key: model_weights.get(key) for key in prev_weights} # these are new weights
        # now wherever the weights returned by server were ALL zeros, attach previous round weights
        for key in self.var_list:
            value = ordered_model_weights.get(key)
            if np.all(value == 0):
                ordered_model_weights[key] = prev_weights[key]

        for key in ordered_model_weights:
            print("ORDER MODEL WEIGHTS: Layer name %s and it's dimension is %r"%(key, len(ordered_model_weights[key])))
            print("Printing the actual matrix %r"%(ordered_model_weights[key]))

        for key in prev_weights:
            print("PREVIOUS MODEL WEIGHTS: Layer name %s and it's dimesion is %r"%(key, len(prev_weights[key])))
            print("Printing the actual matrix %r"%(prev_weights[key]))
            

        # update local model weights with received weights
        self.model.set_weights(list(model_weights.values())) 


        # adjust LR or other training time info as needed
        # such as callback in the fit function
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs_per_round,
            validation_data=(self.X_test, self.y_test),
        )

        # report updated weights in shareable
        weights = {self.model.get_layer(index=key).name: value for key, value in enumerate(self.model.get_weights())}
        
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights)
        dxo = DXO(data_kind=DataKind.WEIGHTS, data=weights, meta={})

        self.log_info(fl_ctx, "Local epochs finished. Returning shareable")
        new_shareable = dxo.to_shareable()
        return new_shareable
